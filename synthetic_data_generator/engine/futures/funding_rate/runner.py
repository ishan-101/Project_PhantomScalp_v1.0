"""Runner for Futures Funding Rate Engine."""

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

from synthetic_data_generator.engine.futures.funding_rate.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.funding_rate.funding_extreme_flag_engine import add_funding_extreme_flag
from synthetic_data_generator.engine.futures.funding_rate.funding_oi_stress_engine import add_funding_oi_stress
from synthetic_data_generator.engine.futures.funding_rate.funding_pressure_index_engine import add_funding_pressure_index
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_acceleration_engine import add_funding_rate_acceleration
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_change_engine import add_funding_rate_change
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_engine import add_funding_rate
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_regime_flag_engine import add_funding_rate_regime_flag
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_velocity_engine import add_funding_rate_velocity
from synthetic_data_generator.engine.futures.funding_rate.funding_rate_zscore_engine import add_funding_rate_zscore
from synthetic_data_generator.engine.futures.funding_rate.loader import load_source_parquets
from synthetic_data_generator.engine.futures.funding_rate.validator import validate_funding_rate_df

ENGINE_NAME = "fut_funding_rate"

FEATURE_COLUMNS = [
    "fut__funding_rate",
    "fut__funding_rate_change",
    "fut__funding_rate_velocity",
    "fut__funding_rate_acceleration",
    "fut__funding_rate_zscore",
    "fut__funding_pressure_index",
    "fut__funding_extreme_flag",
    "fut__funding_oi_stress",
    "fut__funding_rate_regime_flag",
]

FLOAT32_FEATURE_COLUMNS = [
    "fut__funding_rate",
    "fut__funding_rate_change",
    "fut__funding_rate_velocity",
    "fut__funding_rate_acceleration",
    "fut__funding_rate_zscore",
    "fut__funding_pressure_index",
    "fut__funding_oi_stress",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_funding_rate.parquet"


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(
        trades_df=loaded["trades_df"],
        orderflow_df=loaded["orderflow_df"],
        oi_df=loaded["oi_df"],
    )

    df = add_funding_rate(cleaned["trades_df"], cleaned["orderflow_df"], cleaned["oi_df"])
    df = add_funding_rate_change(df)
    df = add_funding_rate_velocity(df)
    df = add_funding_rate_acceleration(df)
    df = add_funding_rate_zscore(df)
    df = add_funding_pressure_index(df)
    df = add_funding_extreme_flag(df)
    df = add_funding_oi_stress(df)
    df = add_funding_rate_regime_flag(df)

    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    seq = SequenceID(seed=cfg.get("global", {}).get("seed", 0))
    df["meta__sequence_id"] = pd.Series(seq.next_batch(len(df)), dtype="int64")

    # Keep output to official funding schema only.
    df = df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]].copy()

    for col in FLOAT32_FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype("float32")
    df["fut__funding_extreme_flag"] = df["fut__funding_extreme_flag"].astype("bool")
    df["fut__funding_rate_regime_flag"] = pd.to_numeric(df["fut__funding_rate_regime_flag"], errors="coerce").fillna(0).astype("int32")

    validate_funding_rate_df(df)

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__funding_rate": "float32",
            "fut__funding_rate_change": "float32",
            "fut__funding_rate_velocity": "float32",
            "fut__funding_rate_acceleration": "float32",
            "fut__funding_rate_zscore": "float32",
            "fut__funding_pressure_index": "float32",
            "fut__funding_extreme_flag": "bool",
            "fut__funding_oi_stress": "float32",
            "fut__funding_rate_regime_flag": "int32",
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

    source_symbol = "UNKNOWN"
    source_exchange = "UNKNOWN"
    if "symbol" in cleaned["trades_df"].columns:
        source_symbol = str(cleaned["trades_df"]["symbol"].iloc[0])
    if "exchange" in cleaned["trades_df"].columns:
        source_exchange = str(cleaned["trades_df"]["exchange"].iloc[0])

    manifest_entry = manifest_mod.build_manifest_entry(
        engine=ENGINE_NAME,
        dataset_type=ENGINE_NAME,
        symbol=source_symbol,
        exchange=source_exchange,
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
        engine_name="futures_funding_rate_runner",
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
                "oi_source": loaded["oi_path"],
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
