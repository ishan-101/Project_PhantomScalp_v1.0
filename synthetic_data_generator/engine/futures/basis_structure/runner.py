"""Runner for Futures Basis Structure Engine."""

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
from synthetic_data_generator.engine.utils.sequence_id import SequenceID

from synthetic_data_generator.engine.futures.basis_structure.loader import load_source_parquets
from synthetic_data_generator.engine.futures.basis_structure.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.basis_structure.perp_spot_basis_engine import add_perp_spot_basis
from synthetic_data_generator.engine.futures.basis_structure.basis_change_engine import add_basis_change
from synthetic_data_generator.engine.futures.basis_structure.basis_velocity_engine import add_basis_velocity
from synthetic_data_generator.engine.futures.basis_structure.basis_zscore_engine import add_basis_zscore
from synthetic_data_generator.engine.futures.basis_structure.basis_regime_flag_engine import add_basis_regime_flag
from synthetic_data_generator.engine.futures.basis_structure.basis_compression_ratio_engine import add_basis_compression_ratio
from synthetic_data_generator.engine.futures.basis_structure.basis_mean_reversion_score_engine import add_basis_mean_reversion_score
from synthetic_data_generator.engine.futures.basis_structure.validator import validate_basis_structure_df


ENGINE_NAME = "fut_basis_structure"
FEATURE_COLUMNS = [
    "fut__perp_spot_basis",
    "fut__basis_change",
    "fut__basis_velocity",
    "fut__basis_zscore",
    "fut__basis_regime_flag",
    "fut__basis_compression_ratio",
    "fut__basis_mean_reversion_score",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_basis_structure.parquet"


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(
        trades_df=loaded["trades_df"],
        orderflow_df=loaded["orderflow_df"],
        funding_df=loaded["funding_df"],
    )

    df = add_perp_spot_basis(cleaned["trades_df"], cleaned["orderflow_df"], cleaned["funding_df"])
    df = add_basis_change(df)
    df = add_basis_velocity(df)
    df = add_basis_zscore(df)
    df = add_basis_regime_flag(df)
    df = add_basis_compression_ratio(df)
    df = add_basis_mean_reversion_score(df)

    # Preserve funding dependencies needed by mean-reversion and output only canonical basis schema.
    keep_cols = ["meta__timestamp", "meta__sequence_id", "fut__funding_rate", "fut__funding_pressure_index", "fut__funding_oi_stress", *FEATURE_COLUMNS]
    df = df[keep_cols].copy()

    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    seq = SequenceID(seed=cfg.get("global", {}).get("seed", 0))
    df["meta__sequence_id"] = pd.Series(seq.next_batch(len(df)), dtype="int64")

    for col in FEATURE_COLUMNS:
        if col == "fut__basis_regime_flag":
            df[col] = df[col].astype(bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype("float32")

    # enforce zero-null for dependency inputs retained internally during scoring lineage
    for dep in ["fut__funding_rate", "fut__funding_pressure_index", "fut__funding_oi_stress"]:
        df[dep] = pd.to_numeric(df[dep], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype("float32")

    validate_basis_structure_df(df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]])

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__perp_spot_basis": "float32",
            "fut__basis_change": "float32",
            "fut__basis_velocity": "float32",
            "fut__basis_zscore": "float32",
            "fut__basis_regime_flag": "bool",
            "fut__basis_compression_ratio": "float32",
            "fut__basis_mean_reversion_score": "float32",
        },
        "max_null_ratio": 0.0,
    }
    shared_result = shared_schema_validator.validate_schema(df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]], schema_spec)
    if not shared_result.get("passed", False):
        raise RuntimeError(f"Shared schema validation failed: {shared_result}")

    out_path = _build_output_path(cfg, selected_date)
    writer = ParquetWriter(out_path, compression=cfg.get("writer", {}).get("compression", "snappy"))
    writer.write(df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]], append=True)
    writer.finalize()
    writer_manifest = writer.get_manifest()

    base_path = Path(cfg.get("paths", {}).get("base", "synthetic_data_generator/outputs")).resolve()
    output_path_resolved = Path(writer_manifest["path"]).resolve()
    try:
        relative_output = output_path_resolved.relative_to(base_path)
    except ValueError:
        relative_output = output_path_resolved

    source_symbol = str(cleaned["trades_df"].get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
    source_exchange = str(cleaned["trades_df"].get("exchange", pd.Series(["UNKNOWN"])).iloc[0])

    manifest_entry = manifest_mod.build_manifest_entry(
        engine=ENGINE_NAME,
        dataset_type=ENGINE_NAME,
        symbol=source_symbol,
        exchange=source_exchange,
        row_count=int(len(df)),
        columns=["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS],
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
        engine_name="futures_basis_structure_runner",
        engine_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_version=cfg.get("meta", {}).get("config_version", "unknown"),
        config_hash=cfg.get("_config_hash") or prov.hash_config(cfg),
        time_range_start=str(df["meta__timestamp"].min()),
        time_range_end=str(df["meta__timestamp"].max()),
        rows=int(len(df)),
        symbol=source_symbol,
        exchange=source_exchange,
        environment=global_cfg.get("environment", "dev"),
        seed=global_cfg.get("seed"),
        notes=json.dumps(
            {
                "trades_source": loaded["trades_path"],
                "orderflow_source": loaded["orderflow_path"],
                "funding_source": loaded["funding_path"],
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
