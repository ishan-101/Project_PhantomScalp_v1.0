#!/usr/bin/env python3
"""
ticks_trades_engine.py
Produces a trades parquet file with columns:
  meta__timestamp (tz-aware UTC), meta__sequence_id, price, size, aggressor, exchange, symbol

Design goals:
- Defensive: will use your project's SequenceID and RNG if present.
- Writes to: data/synthetic_data/ticks_and_orderflow/trades/date=YYYY-MM-DD/ticks_trades.parquet (default).
- Returns a result dict similar to other runners.
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

# try using project utilities if available
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
    """Generic wrapper: try common method names on seq_obj instance."""
    if seq_obj is None:
        # fallback to numpy ints
        return list(np.arange(1, n+1, dtype=np.int64))
    inst = seq_obj()
    for method in ("next_id", "next", "nextval", "next_id_int", "get_next"):
        if hasattr(inst, method):
            fn = getattr(inst, method)
            return [int(fn()) for _ in range(n)]
    # last fallback: if instance is callable and returns increasing ints
    try:
        return [int(inst()) for _ in range(n)]
    except Exception:
        # generate monotonic ids
        start = int(dt.datetime.utcnow().timestamp() * 1_000_000) % (2**60)
        return list(np.arange(start, start+n, dtype=np.int64))

def run_engine(cfg: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    """
    Generate trades data and write parquet.

    cfg (optional) may contain:
      - n: number rows (default 50000)
      - out_dir: explicit output directory (overrides config/default)
      - out_base: base output directory (config-aware; engine will append partition)
      - date: partition date string (YYYY-MM-DD) default today
      - compression
    """
    cfg = cfg or {}
    central_cfg = _load_central_config()

    # rows default: prefer config rows.ticks_trades
    default_n = 50000
    if isinstance(central_cfg, dict):
        try:
            default_n = int(central_cfg.get("rows", {}).get("ticks_trades", default_n))
        except Exception:
            pass
    n = int(cfg.get("n", default_n))

    date_str = cfg.get("date") or dt.datetime.utcnow().strftime("%Y-%m-%d")

    # resolve output directory using config paths if available
    paths_cfg = central_cfg.get("paths", {}) if isinstance(central_cfg, dict) else {}
    base_path = pathlib.Path(paths_cfg.get("base", "synthetic_data_generator/outputs"))
    sub_path = pathlib.Path(paths_cfg.get("ticks_trades", "ticks_and_orderflow/trades"))

    if "out_dir" in cfg:
        out_dir = pathlib.Path(cfg["out_dir"])
    elif "out_base" in cfg:
        # preserve backward compatibility: out_base is root before appending partition
        out_dir = pathlib.Path(cfg["out_base"]) / f"date={date_str}"
    else:
        out_dir = base_path / sub_path / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ticks_trades.parquet"

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

    # generate columns
    # timestamps: evenly spaced inside one trading day (UTC), with microsecond uniqueness
    start_ts = pd.Timestamp.utcnow().tz_convert("UTC") if hasattr(pd.Timestamp, "tz_convert") else pd.Timestamp.utcnow().tz_localize("UTC")
    # create monotonic microsecond offsets
    offsets_us = np.arange(0, n * 1000, step=1000, dtype=np.int64)  # 1ms apart default
    timestamps = (start_ts + pd.to_timedelta(offsets_us, unit="us"))
    # sequence ids
    SequenceID = _import_sequence_id()
    seq = _next_seq_ids(SequenceID, n)

    # price / size / aggressor / exchange / symbol
    if rng is not None:
        try:
            price = rng.normal("price", loc=30000.0, scale=100.0, size=n)
            size = rng.uniform("size", low=0.01, high=5.0, size=n)
            aggressor = rng.choice("aggr", [-1, 1], size=n)
            exchange = rng.choice("exchange", ["EX1", "EX2"], size=n)
            symbol = rng.choice("symbol", ["BTCUSD"], size=n)
        except Exception:
            rng = None

    if rng is None:
        price = np.random.normal(loc=30000.0, scale=100.0, size=n)
        size = np.random.uniform(low=0.01, high=5.0, size=n)
        aggressor = np.random.choice([-1, 1], size=n)
        exchange = np.random.choice(["EX1", "EX2"], size=n)
        symbol = np.array(["BTCUSD"] * n)

    df = pd.DataFrame({
        "meta__timestamp": timestamps,
        "meta__sequence_id": np.array(seq, dtype="int64"),
        "price": price.astype(float),
        "size": size.astype(float),
        "aggressor": aggressor.astype(int),
        "exchange": exchange,
        "symbol": symbol
    })

    compression = cfg.get("compression", compression_default)

    # write via io_writer if available (it handles meta columns), else pandas
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
