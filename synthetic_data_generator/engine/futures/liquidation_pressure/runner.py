"""Runner for Futures Liquidation Pressure Engine."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
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

from synthetic_data_generator.engine.futures.liquidation_pressure.loader import load_source_parquets
from synthetic_data_generator.engine.futures.liquidation_pressure.cleaner import clean_source_frames
from synthetic_data_generator.engine.futures.liquidation_pressure.long_liquidation_volume_engine import add_long_liquidation_volume
from synthetic_data_generator.engine.futures.liquidation_pressure.short_liquidation_volume_engine import add_short_liquidation_volume
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_imbalance_engine import add_liquidation_imbalance
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_cluster_distance_engine import add_liquidation_cluster_distance
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_velocity_engine import add_liquidation_velocity
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_pressure_index_engine import add_liquidation_pressure_index
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_cascade_probability_engine import add_liquidation_cascade_probability
from synthetic_data_generator.engine.futures.liquidation_pressure.liquidation_heat_pressure_engine import add_liquidation_heat_pressure
from synthetic_data_generator.engine.futures.liquidation_pressure.validator import validate_liquidation_pressure_df


ENGINE_NAME = "fut_liquidation_pressure"
FEATURE_COLUMNS = [
    "fut__long_liquidation_volume",
    "fut__short_liquidation_volume",
    "fut__liquidation_imbalance",
    "fut__liquidation_cluster_distance",
    "fut__liquidation_pressure_index",
    "fut__liquidation_cascade_probability",
    "fut__liquidation_velocity",
    "fut__liquidation_heat_pressure",
]


def _build_output_path(cfg: Dict[str, Any], partition_date: str) -> Path:
    paths_cfg = cfg.get("paths", {})
    base = Path(paths_cfg["base"])
    rel = Path(paths_cfg[ENGINE_NAME])
    out_dir = base / rel / f"date={partition_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fut_liquidation_pressure.parquet"


def _build_market_timeline(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame) -> pd.DataFrame:
    tr = trades_df[["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]].copy()
    tr["meta__timestamp"] = pd.to_datetime(tr["meta__timestamp"], utc=True)
    tr["meta__sequence_id"] = pd.to_numeric(tr["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce").fillna(0.0)
    tr["size"] = pd.to_numeric(tr["size"], errors="coerce").fillna(0.0)
    tr["aggressor"] = pd.to_numeric(tr["aggressor"], errors="coerce").fillna(0.0)

    tr["__liq__buy_aggr_notional"] = np.where(tr["aggressor"] > 0, tr["price"] * tr["size"], 0.0)
    tr["__liq__sell_aggr_notional"] = np.where(tr["aggressor"] < 0, tr["price"] * tr["size"], 0.0)
    tr["__liq__price"] = tr["price"]

    of = orderflow_df[["meta__timestamp", "meta__sequence_id", "event_type", "size", "aggressor", "inventory_pressure"]].copy()
    of["meta__timestamp"] = pd.to_datetime(of["meta__timestamp"], utc=True)
    of["meta__sequence_id"] = pd.to_numeric(of["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    of["size"] = pd.to_numeric(of["size"], errors="coerce").fillna(0.0)
    of["aggressor"] = pd.to_numeric(of["aggressor"], errors="coerce").fillna(0.0)
    of["inventory_pressure"] = pd.to_numeric(of["inventory_pressure"], errors="coerce").fillna(0.0)

    et = of["event_type"].astype(str).str.lower()
    trade_impulse = np.where(et.eq("trade"), of["aggressor"] * of["size"], 0.0)
    queue_impulse = np.where(et.eq("add"), of["size"], np.where(et.eq("cancel"), -of["size"], 0.0))
    of["__liq__orderflow_aggression"] = (trade_impulse + 0.20 * queue_impulse + 0.15 * of["inventory_pressure"]).astype("float64")

    of_agg = of.groupby("meta__timestamp", sort=True)["__liq__orderflow_aggression"].mean().rename("__liq__orderflow_aggression")

    timeline = tr[["meta__timestamp", "meta__sequence_id", "__liq__price", "__liq__buy_aggr_notional", "__liq__sell_aggr_notional"]].copy()
    timeline = timeline.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)
    timeline = timeline.merge(of_agg, on="meta__timestamp", how="left")
    timeline["__liq__orderflow_aggression"] = pd.to_numeric(timeline["__liq__orderflow_aggression"], errors="coerce").fillna(0.0)

    timeline["__liq__price_return"] = pd.to_numeric(timeline["__liq__price"], errors="coerce").pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    vpin_num = (timeline["__liq__buy_aggr_notional"] - timeline["__liq__sell_aggr_notional"]).abs().rolling(120, min_periods=5).sum()
    vpin_den = (timeline["__liq__buy_aggr_notional"] + timeline["__liq__sell_aggr_notional"]).rolling(120, min_periods=5).sum().replace(0.0, np.nan)
    timeline["__liq__toxicity_proxy"] = (vpin_num / vpin_den).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return timeline


def _attach_dependencies(timeline: pd.DataFrame, oi_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    oi_cols = ["meta__timestamp", "fut__open_interest", "fut__oi_velocity", "fut__oi_zscore"]
    funding_cols = ["meta__timestamp", "fut__funding_rate_zscore", "fut__funding_oi_stress", "fut__funding_pressure_index"]

    oi_map = oi_df[oi_cols].copy().sort_values("meta__timestamp")
    funding_map = funding_df[funding_cols].copy().sort_values("meta__timestamp")

    merged = pd.merge_asof(
        timeline.sort_values("meta__timestamp"),
        oi_map,
        on="meta__timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = pd.merge_asof(
        merged.sort_values("meta__timestamp"),
        funding_map,
        on="meta__timestamp",
        direction="backward",
        allow_exact_matches=True,
    )

    fill_cols = [
        "fut__open_interest",
        "fut__oi_velocity",
        "fut__oi_zscore",
        "fut__funding_rate_zscore",
        "fut__funding_oi_stress",
        "fut__funding_pressure_index",
    ]
    merged[fill_cols] = merged[fill_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    # OI build density proxy for cluster mapping.
    oi_level = pd.to_numeric(merged["fut__open_interest"], errors="coerce").fillna(0.0)
    oi_delta = oi_level.diff().fillna(0.0)
    merged["__liq__oi_build_density"] = (
        (oi_delta.clip(lower=0.0)).rolling(180, min_periods=5).sum()
        / oi_level.rolling(180, min_periods=5).mean().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return merged


def run_engine(partition_date: str | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config()

    loaded = load_source_parquets(partition_date=partition_date)
    selected_date = str(loaded["partition_date"])

    cleaned = clean_source_frames(
        trades_df=loaded["trades_df"],
        orderflow_df=loaded["orderflow_df"],
        oi_df=loaded["oi_df"],
        funding_df=loaded["funding_df"],
    )

    df = _build_market_timeline(cleaned["trades_df"], cleaned["orderflow_df"])
    df = _attach_dependencies(df, cleaned["oi_df"], cleaned["funding_df"])

    df = add_long_liquidation_volume(df)
    df = add_short_liquidation_volume(df)
    df = add_liquidation_imbalance(df)
    df = add_liquidation_cluster_distance(df)
    df = add_liquidation_velocity(df)
    df = add_liquidation_pressure_index(df)
    df = add_liquidation_cascade_probability(df)
    df = add_liquidation_heat_pressure(df)

    df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    seq = SequenceID(seed=cfg.get("global", {}).get("seed", 0))
    df["meta__sequence_id"] = pd.Series(seq.next_batch(len(df)), dtype="int64")

    df = df[["meta__timestamp", "meta__sequence_id", *FEATURE_COLUMNS]].copy()
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    validate_liquidation_pressure_df(df)

    schema_spec = {
        "required_columns": {
            "meta__timestamp": "datetime64[ns, UTC]",
            "meta__sequence_id": "int64",
            "fut__long_liquidation_volume": "float32",
            "fut__short_liquidation_volume": "float32",
            "fut__liquidation_imbalance": "float32",
            "fut__liquidation_cluster_distance": "float32",
            "fut__liquidation_pressure_index": "float32",
            "fut__liquidation_cascade_probability": "float32",
            "fut__liquidation_velocity": "float32",
            "fut__liquidation_heat_pressure": "float32",
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

    source_symbol = str(cleaned["trades_df"].get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
    source_exchange = str(cleaned["trades_df"].get("exchange", pd.Series(["UNKNOWN"])).iloc[0])

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
        engine_name="futures_liquidation_pressure_runner",
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
