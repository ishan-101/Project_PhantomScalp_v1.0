# orderbook_l2_engine.py
from __future__ import annotations
import time
import math
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import logging

from synthetic_data_generator.engine.utils.io_writer import ParquetWriter
from synthetic_data_generator.engine.utils.rng import RNG
from synthetic_data_generator.engine.utils.clock import CanonicalClock
from synthetic_data_generator.engine.utils.sequence_id import SequenceID

log = logging.getLogger("orderbook_l2")
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[orderbook_l2] %(message)s"))
    log.addHandler(ch)
log.setLevel(logging.INFO)

DEFAULT_CONFIG = {
    "rows": 250_000,
    "batch_size": 20_000,
    "out_path": "data/synthetic_data/orderbook/l2/orderbook_l2.parquet",
    "seed": 9999,
    "compression": "snappy",
    # snapshot parameters
    "levels": 10,
    "price_mu": 30000.0,
    "price_sigma": 50.0,
}


def _make_out_path(cfg: Dict, date_str: str) -> Path:
    base = Path(cfg.get("out_path", DEFAULT_CONFIG["out_path"]))
    out_dir = base.parent / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / base.name


def _build_snapshot(center_price: float, rng: RNG, levels: int):
    """
    Build a single L2 snapshot row (flattened columns).
    """
    # build bids (descending) and asks (ascending)
    bids = []
    asks = []
    bid_sizes = []
    ask_sizes = []

    # level spacing: small random tick distances
    for lvl in range(levels):
        tick = lvl + float(rng.uniform(low=0.1, high=1.0))
        bids.append(center_price - tick)
        asks.append(center_price + tick)
        bid_sizes.append(abs(float(rng.normal(loc=1.0, scale=0.5))) + 1e-8)
        ask_sizes.append(abs(float(rng.normal(loc=1.0, scale=0.5))) + 1e-8)

    top_bid = bids[0]
    top_ask = asks[0]
    spread = float(top_ask - top_bid)

    row = {
        "top_bid": float(top_bid),
        "top_ask": float(top_ask),
        "spread": spread,
    }

    for i in range(levels):
        row[f"bid_price_{i}"] = float(bids[i])
        row[f"bid_size_{i}"] = float(bid_sizes[i])
        row[f"ask_price_{i}"] = float(asks[i])
        row[f"ask_size_{i}"] = float(ask_sizes[i])

    return row


def run_engine(cfg: Dict | None = None) -> Dict:
    cfg = cfg or DEFAULT_CONFIG
    rows_target = int(cfg["rows"])
    batch_size = int(cfg["batch_size"])
    out_path = _make_out_path(cfg, pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    seed = int(cfg["seed"])
    compression = cfg.get("compression", "snappy")
    levels = int(cfg.get("levels", 10))

    rng = RNG(seed)
    seq = SequenceID(start=0)
    clock = CanonicalClock(config={"seed": seed})

    writer = ParquetWriter(out_path, compression=compression)

    total = 0
    started = time.time()
    batch_idx = 0

    try:
        while total < rows_target:
            n = min(batch_size, rows_target - total)

            timestamps = [clock.now() for _ in range(n)]
            seq_ids = [int(seq.next()) for _ in range(n)]
            centers = rng.normal(loc=cfg["price_mu"], scale=cfg["price_sigma"], size=n)

            rows = []
            for i in range(n):
                snap = _build_snapshot(float(centers[i]), rng, levels)
                snap.update(
                    {
                        "meta__timestamp": timestamps[i],
                        "meta__sequence_id": seq_ids[i],
                        "exchange": "EXCH",
                        "symbol": "BTCUSD",
                    }
                )
                rows.append(snap)

            df = pd.DataFrame(rows)
            # ensure timezone-aware timestamps
            df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True)

            writer.write(df, append=True)
            total += n
            batch_idx += 1
            elapsed = time.time() - started
            log.info(f"batch {batch_idx}: wrote {min(total, rows_target)} snapshots — elapsed {elapsed:.1f}s")

        writer.finalize()
        manifest = writer.get_manifest()
        return {"rows": total, "manifest": manifest, "timing": {"seconds": time.time() - started}}
    except Exception:
        log.exception("Engine raised exception:")
        try:
            writer.finalize()
        except Exception:
            log.exception("Finalize also failed.")
        raise


if __name__ == "__main__":
    log.info("Starting orderbook_l2 engine...")
    res = run_engine()
    log.info("Done.")
    print(res)
