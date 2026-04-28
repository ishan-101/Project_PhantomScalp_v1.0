"""Runner for Futures Volume Flow Engine."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
import os
import sys

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

from synthetic_data_generator.engine.futures.volume_flow.loader import load_source_parquets
from synthetic_data_generator.engine.futures.volume_flow.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.volume_flow.trade_volume_engine import add_trade_volume
from synthetic_data_generator.engine.futures.volume_flow.volume_delta_engine import add_volume_delta
from synthetic_data_generator.engine.futures.volume_flow.volume_delta_ratio_engine import add_volume_delta_ratio
from synthetic_data_generator.engine.futures.volume_flow.large_trade_volume_engine import add_large_trade_volume
from synthetic_data_generator.engine.futures.volume_flow.volume_burst_intensity_engine import add_volume_burst_intensity
from synthetic_data_generator.engine.futures.volume_flow.validator import validate_volume_flow_df


ENGINE_NAME = "fut_volume_flow"
FEATURE_COLUMNS = [
    "fut__trade_volume",
    "fut__volume_delta",
    "fut__volume_delta_ratio",
    "fut__large_trade_volume",
    "fut__volume_burst_intensity",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_volume_flow.parquet"


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(
        trades_df=loaded["trades_df"],
        orderflow_df=loaded["orderflow_df"],
    )

    trades_df = cleaned["trades_df"].copy(deep=True)

    df = add_trade_volume(trades_df)
    df = add_volume_delta(df)
    df = add_volume_delta_ratio(df)
    df = add_large_trade_volume(df)
    df = add_volume_burst_intensity(df, cleaned["orderflow_df"])

    # Canonical output schema only.
    df = df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]].copy()

    # Enforce canonical dtypes and rebuild sequence IDs.
    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    seq = SequenceID(seed=cfg.get("global", {}).get("seed", 0))
    df["meta__sequence_id"] = pd.Series(seq.next_batch(len(df)), dtype="int64")

    for col in FEATURE_COLUMNS:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
            .astype("float32")
        )

    validate_volume_flow_df(df)

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__trade_volume": "float32",
            "fut__volume_delta": "float32",
            "fut__volume_delta_ratio": "float32",
            "fut__large_trade_volume": "float32",
            "fut__volume_burst_intensity": "float32",
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
        engine_name="futures_volume_flow_runner",
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
