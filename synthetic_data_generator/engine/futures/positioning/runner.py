"""Runner for Futures Positioning Engine."""

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

from synthetic_data_generator.engine.futures.positioning.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.positioning.loader import load_source_parquets
from synthetic_data_generator.engine.futures.positioning.long_short_ratio_engine import add_long_short_ratio
from synthetic_data_generator.engine.futures.positioning.long_short_ratio_change_engine import add_long_short_ratio_change
from synthetic_data_generator.engine.futures.positioning.net_long_position_proxy_engine import add_net_long_position_proxy
from synthetic_data_generator.engine.futures.positioning.net_short_position_proxy_engine import add_net_short_position_proxy
from synthetic_data_generator.engine.futures.positioning.position_skew_engine import add_position_skew
from synthetic_data_generator.engine.futures.positioning.net_position_change_velocity_engine import add_net_position_change_velocity
from synthetic_data_generator.engine.futures.positioning.validator import validate_positioning_df


ENGINE_NAME = "fut_positioning"
FEATURE_COLUMNS = [
    "fut__long_short_ratio",
    "fut__long_short_ratio_change",
    "fut__net_long_position_proxy",
    "fut__net_short_position_proxy",
    "fut__position_skew",
    "fut__net_position_change_velocity",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_positioning.parquet"


def _align_dependencies(oi_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["meta__timestamp", "meta__sequence_id"]
    merged = pd.merge(
        oi_df[join_cols + ["fut__open_interest", "fut__oi_zscore", "fut__oi_change"]],
        funding_df[
            join_cols
            + [
                "fut__funding_rate",
                "fut__funding_rate_zscore",
                "fut__funding_rate_regime_flag",
                "fut__funding_oi_stress",
            ]
        ]
        if "fut__funding_oi_stress" in funding_df.columns
        else funding_df[
            join_cols
            + [
                "fut__funding_rate",
                "fut__funding_rate_zscore",
                "fut__funding_rate_regime_flag",
                "fut__funding_stress_score",
            ]
        ],
        on=join_cols,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError("Dependency alignment failed: no overlapping rows between open_interest and funding_rate")

    if "fut__funding_oi_stress" in merged.columns:
        merged["__pos__funding_oi_stress"] = pd.to_numeric(merged["fut__funding_oi_stress"], errors="coerce").fillna(0.0)
    else:
        merged["__pos__funding_oi_stress"] = pd.to_numeric(merged["fut__funding_stress_score"], errors="coerce").fillna(0.0)

    return merged.sort_values(join_cols, kind="mergesort").reset_index(drop=True)


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(oi_df=loaded["oi_df"], funding_df=loaded["funding_df"])
    df = _align_dependencies(cleaned["oi_df"], cleaned["funding_df"])

    df = add_long_short_ratio(df)
    df = add_long_short_ratio_change(df)
    df = add_net_long_position_proxy(df)
    df = add_net_short_position_proxy(df)
    df = add_position_skew(df)
    df = add_net_position_change_velocity(df)

    # canonical meta
    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    seq = SequenceID(seed=cfg.get("global", {}).get("seed", 0))
    df["meta__sequence_id"] = pd.Series(seq.next_batch(len(df)), dtype="int64")

    df = df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]].copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype("float32")

    validate_positioning_df(df)

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__long_short_ratio": "float32",
            "fut__long_short_ratio_change": "float32",
            "fut__net_long_position_proxy": "float32",
            "fut__net_short_position_proxy": "float32",
            "fut__position_skew": "float32",
            "fut__net_position_change_velocity": "float32",
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
        engine_name="futures_positioning_runner",
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
                "oi_source": loaded["oi_path"],
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
