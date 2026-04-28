"""Runner for Futures Leverage Metrics Engine."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_here = os.path.dirname(__file__)
root = os.path.abspath(os.path.join(_here, "../../../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from synthetic_data_generator.engine.config.loader import load_config
from synthetic_data_generator.engine.meta_provenance import manifest as manifest_mod
from synthetic_data_generator.engine.meta_provenance import provenance_helper as prov
from synthetic_data_generator.engine.meta_provenance import schema_validator as shared_schema_validator
from synthetic_data_generator.engine.utils.io_writer import ParquetWriter

from synthetic_data_generator.engine.futures.leverage_metrics.loader import load_source_parquets
from synthetic_data_generator.engine.futures.leverage_metrics.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.leverage_metrics.oi_volume_interaction_engine import add_oi_volume_interaction
from synthetic_data_generator.engine.futures.leverage_metrics.funding_oi_interaction_engine import add_funding_oi_interaction
from synthetic_data_generator.engine.futures.leverage_metrics.basis_oi_interaction_engine import add_basis_oi_interaction
from synthetic_data_generator.engine.futures.leverage_metrics.leverage_pressure_index_engine import add_leverage_pressure_index
from synthetic_data_generator.engine.futures.leverage_metrics.validator import validate_leverage_metrics_df


ENGINE_NAME = "fut_leverage_metrics"
FEATURE_COLUMNS = [
    "fut__oi_volume_interaction",
    "fut__funding_oi_interaction",
    "fut__basis_oi_interaction",
    "fut__leverage_pressure_index",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_leverage_metrics.parquet"


def _build_feature_frame(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Anchor on the lower-frequency futures interaction clock (basis/volume) and join
    # higher-frequency dependencies by canonical deterministic sequence id.
    base = cleaned["basis_df"][["meta__timestamp", "meta__sequence_id", "fut__basis_zscore"]].copy()

    df = base.merge(
        cleaned["volume_flow_df"][["meta__sequence_id", "fut__volume_delta_ratio"]],
        on="meta__sequence_id",
        how="inner",
        sort=False,
    )
    df = df.merge(
        cleaned["oi_df"][["meta__sequence_id", "fut__oi_zscore"]],
        on="meta__sequence_id",
        how="inner",
        sort=False,
    )
    df = df.merge(
        cleaned["funding_df"][["meta__sequence_id", "fut__funding_rate_zscore", "fut__funding_oi_stress"]],
        on="meta__sequence_id",
        how="inner",
        sort=False,
    )

    if df.empty:
        raise RuntimeError("Merged dependency frame is empty; no overlap across required sources")

    df = df.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)
    return df


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(
        oi_df=loaded["oi_df"],
        funding_df=loaded["funding_df"],
        basis_df=loaded["basis_df"],
        volume_flow_df=loaded["volume_flow_df"],
    )

    df = _build_feature_frame(cleaned)
    df = add_oi_volume_interaction(df)
    df = add_funding_oi_interaction(df)
    df = add_basis_oi_interaction(df)
    df = add_leverage_pressure_index(df)

    df = df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]].copy()
    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    df["meta__sequence_id"] = pd.to_numeric(df["meta__sequence_id"], errors="coerce").astype("int64")
    for col in FEATURE_COLUMNS:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
            .astype("float32")
        )

    validate_leverage_metrics_df(df)

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__oi_volume_interaction": "float32",
            "fut__funding_oi_interaction": "float32",
            "fut__basis_oi_interaction": "float32",
            "fut__leverage_pressure_index": "float32",
        },
        "max_null_ratio": 0.0,
    }
    shared_result = shared_schema_validator.validate_schema(df, schema_spec)
    if not shared_result.get("passed", False):
        raise RuntimeError(f"Shared schema validation failed: {shared_result}")

    out_path = _build_output_path(cfg, selected_date)
    writer = ParquetWriter(out_path, compression=cfg.get("writer", {}).get("compression", "snappy"))
    writer.write(df, append=True)
    writer.finalize()
    writer_manifest = writer.get_manifest()

    base_path = Path(cfg.get("paths", {}).get("base", "synthetic_data_generator/outputs")).resolve()
    output_path_resolved = Path(writer_manifest["path"]).resolve()
    try:
        relative_output = output_path_resolved.relative_to(base_path)
    except ValueError:
        relative_output = output_path_resolved

    manifest_entry = manifest_mod.build_manifest_entry(
        engine=ENGINE_NAME,
        dataset_type=ENGINE_NAME,
        symbol="UNKNOWN",
        exchange="UNKNOWN",
        row_count=int(len(df)),
        columns=list(df.columns),
        partition_date=selected_date,
        file_path=output_path_resolved,
        config=cfg,
    )
    manifest_entry["file_path"] = str(relative_output)

    manifest_cfg = cfg.get("manifest", {})
    manifest_name = manifest_cfg.get("manifest_name", "_manifest.json")
    manifest_dir = Path(cfg.get("paths", {}).get("meta", "meta"))
    manifest_path = base_path / manifest_dir / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_mod.append_manifest(manifest_entry, manifest_path)

    global_cfg = cfg.get("global", {})
    provenance_record = prov.build_provenance_record(
        dataset_name=ENGINE_NAME,
        engine_name="futures_leverage_metrics_runner",
        engine_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_hash=cfg.get("_config_hash") or prov.hash_config(cfg),
        time_range_start=str(df["meta__timestamp"].min()),
        time_range_end=str(df["meta__timestamp"].max()),
        rows=int(len(df)),
        symbol="UNKNOWN",
        exchange="UNKNOWN",
        environment=global_cfg.get("environment", "dev"),
        seed=global_cfg.get("seed"),
        notes=json.dumps(
            {
                "open_interest_source": loaded["oi_path"],
                "funding_source": loaded["funding_path"],
                "basis_source": loaded["basis_path"],
                "volume_flow_source": loaded["volume_flow_path"],
                "output_path": str(relative_output),
            }
        ),
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "engine": ENGINE_NAME,
        "rows": int(len(df)),
        "partition_date": selected_date,
        "output_files": [str(relative_output)],
        "manifest": manifest_entry,
        "provenance": provenance_record,
        "timing_ms": elapsed_ms,
    }


if __name__ == "__main__":
    print(json.dumps(run_engine(), indent=2, default=str))
