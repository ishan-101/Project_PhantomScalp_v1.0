#!/usr/bin/env python3
"""
ticks_orderflow_engine.py
Produces orderflow parquet with columns:
  meta__timestamp (tz-aware UTC), meta__sequence_id, event_type, price, size, aggressor, inventory_pressure, exchange, symbol

Defaults:
  n = 500000
  out: data/synthetic_data/ticks_and_orderflow/orderflow/date=YYYY-MM-DD/ticks_orderflow.parquet
"""
import os
import pathlib
import datetime as dt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def _load_central_config() -> Optional[Dict[str, Any]]:
    """
    Best-effort config loader. Tries the shared loader first, otherwise returns None.
    """
    for mod_path in (
        "synthetic_data_generator.engine.config.loader",
        "engine.config.loader",
    ):
        try:
            mod = __import__(mod_path, fromlist=["load_config"])
            if hasattr(mod, "load_config"):
                cfg = mod.load_config()
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            continue
    return None

def _import_sequence_id():
    try:
        from synthetic_data_generator.engine.utils.sequence_id import SequenceID
    except Exception:
        try:
            from engine.utils.sequence_id import SequenceID
        except Exception:
            SequenceID = None
    return SequenceID

def _import_rng():
    try:
        from synthetic_data_generator.engine.utils.rng import RNG
    except Exception:
        try:
            from engine.utils.rng import RNG
        except Exception:
            RNG = None
    return RNG

def _import_io_writer():
    try:
        from synthetic_data_generator.engine.utils.io_writer import ParquetWriter
    except Exception:
        try:
            from engine.utils.io_writer import ParquetWriter
        except Exception:
            ParquetWriter = None
    return ParquetWriter

def _next_seq_ids(seq_obj, n):
    if seq_obj is None:
        return list(np.arange(1, n+1, dtype=np.int64))
    inst = seq_obj()
    for method in ("next_id", "next", "nextval", "get_next", "next_id_int"):
        if hasattr(inst, method):
            fn = getattr(inst, method)
            return [int(fn()) for _ in range(n)]
    try:
        return [int(inst()) for _ in range(n)]
    except Exception:
        start = int(dt.datetime.utcnow().timestamp() * 1_000_000) % (2**60)
        return list(np.arange(start, start+n, dtype=np.int64))

def run_engine(cfg: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    cfg = cfg or {}
    central_cfg = _load_central_config()

    # rows default from config if available
    default_n = 500000
    if isinstance(central_cfg, dict):
        try:
            default_n = int(central_cfg.get("rows", {}).get("ticks_orderflow", default_n))
        except Exception:
            pass
    n = int(cfg.get("n", default_n))

    date_str = cfg.get("date") or dt.datetime.utcnow().strftime("%Y-%m-%d")

    # resolve output directory using config paths if available
    paths_cfg = central_cfg.get("paths", {}) if isinstance(central_cfg, dict) else {}
    base_path = pathlib.Path(paths_cfg.get("base", "synthetic_data_generator/outputs"))
    sub_path = pathlib.Path(paths_cfg.get("ticks_orderflow", "ticks_and_orderflow/orderflow"))

    if "out_dir" in cfg:
        out_dir = pathlib.Path(cfg["out_dir"])
    elif "out_base" in cfg:
        # backward compatible: out_base is root before appending partition
        out_dir = pathlib.Path(cfg["out_base"]) / f"date={date_str}"
    else:
        out_dir = base_path / sub_path / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ticks_orderflow.parquet"

    RNG = _import_rng()
    if RNG is not None:
        try:
            rng = RNG()
        except Exception:
            rng = None
    else:
        rng = None

    # compression preference
    compression_default = "snappy"
    if isinstance(central_cfg, dict):
        try:
            compression_default = central_cfg.get("writer", {}).get("compression", compression_default)
        except Exception:
            pass

    ParquetWriter = _import_io_writer()
    SequenceID = _import_sequence_id()
    seq = _next_seq_ids(SequenceID, n)

    # timestamps
    start_ts = pd.Timestamp.utcnow().tz_convert("UTC") if hasattr(pd.Timestamp, "tz_convert") else pd.Timestamp.utcnow().tz_localize("UTC")
    offsets_us = np.arange(0, n * 500, step=500, dtype=np.int64)  # 0.5ms spacing
    timestamps = (start_ts + pd.to_timedelta(offsets_us, unit="us"))

    if rng is not None:
        try:
            event_type = rng.choice("event_type", ["add", "cancel", "trade"], size=n, p=[0.6, 0.2, 0.2])
            price = rng.normal("price", loc=30000.0, scale=120.0, size=n)
            size = rng.uniform("size", low=0.001, high=10.0, size=n)
            aggressor = rng.choice("aggr", [-1, 0, 1], size=n, p=[0.45, 0.1, 0.45])
            inv_pressure = rng.normal("inv_pressure", loc=0.0, scale=1.0, size=n)
            exchange = rng.choice("exchange", ["EX1","EX2"], size=n)
            symbol = rng.choice("symbol", ["BTCUSD"], size=n)
        except Exception:
            rng = None

    if rng is None:
        event_type = np.random.choice(["add","cancel","trade"], size=n, p=[0.6,0.2,0.2])
        price = np.random.normal(loc=30000.0, scale=120.0, size=n)
        size = np.random.uniform(low=0.001, high=10.0, size=n)
        aggressor = np.random.choice([-1,0,1], size=n, p=[0.45,0.1,0.45])
        inv_pressure = np.random.normal(loc=0.0, scale=1.0, size=n)
        exchange = np.random.choice(["EX1","EX2"], size=n)
        symbol = np.array(["BTCUSD"] * n)

    df = pd.DataFrame({
        "meta__timestamp": timestamps,
        "meta__sequence_id": np.array(seq, dtype="int64"),
        "event_type": event_type,
        "price": price.astype(float),
        "size": size.astype(float),
        "aggressor": aggressor.astype(int),
        "inventory_pressure": inv_pressure.astype(float),
        "exchange": exchange,
        "symbol": symbol
    })

    compression = cfg.get("compression", compression_default)

    if ParquetWriter is not None:
        try:
            writer = ParquetWriter(str(out_path), compression=compression)
            writer.write(df)
            writer.finalize()
        except Exception:
            df.to_parquet(str(out_path), index=False, compression=compression)
    else:
        df.to_parquet(str(out_path), index=False, compression=compression)

    return {
        "rows": int(len(df)),
        "manifest": {
            "path": str(out_path),
            "rows": int(len(df)),
            "bytes": int(os.path.getsize(str(out_path))) if out_path.exists() else 0
        }
    }

if __name__ == "__main__":
    import json, sys
    res = run_engine()
    print(json.dumps(res, indent=2))
