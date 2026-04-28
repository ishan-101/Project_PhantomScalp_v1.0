# orderbook_l3_engine.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import logging

from synthetic_data_generator.engine.utils.io_writer import ParquetWriter
from synthetic_data_generator.engine.utils.rng import RNG
from synthetic_data_generator.engine.utils.clock import CanonicalClock
from synthetic_data_generator.engine.utils.sequence_id import SequenceID

log = logging.getLogger("orderbook_l3")
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[orderbook_l3] %(message)s"))
    log.addHandler(ch)
log.setLevel(logging.INFO)

DEFAULT_CONFIG = {
    "rows": 300_000,
    "part_rows": 20_000,
    "out_path": "data/synthetic_data/orderbook/l3/orderbook_l3.parquet",
    "seed": 12345,
    "compression": "snappy",
    # pace parameters (kept as tunables)
    "price_mu": 30000.0,
    "price_sigma": 200.0,
    "size_mu": 0.5,
    "size_sigma": 1.0,
}


def _make_out_path(cfg: Dict, date_str: str) -> Path:
    base = Path(cfg.get("out_path", DEFAULT_CONFIG["out_path"]))
    # prefer partitioned path under date=YYYY-MM-DD
    out_dir = base.parent / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / base.name


def run_engine(cfg: Dict | None = None) -> Dict:
    cfg = cfg or DEFAULT_CONFIG
    rows_target = int(cfg["rows"])
    part_rows = int(cfg["part_rows"])
    out_path = _make_out_path(cfg, pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    seed = int(cfg["seed"])
    compression = cfg.get("compression", "snappy")

    rng = RNG(seed)
    seq = SequenceID(start=0)
    clock = CanonicalClock(config={"seed": seed})

    writer = ParquetWriter(out_path, compression=compression)

    total = 0
    started = time.time()
    part_idx = 0

    try:
        while total < rows_target:
            n = min(part_rows, rows_target - total)

            # timestamps (timezone-aware)
            timestamps = [clock.now() for _ in range(n)]

            # event types and sides
            event_types = rng.choice(["add", "cancel", "trade"], size=n, p=[0.6, 0.2, 0.2])
            sides = rng.choice(["B", "S"], size=n)

            # prices and sizes
            prices = rng.normal(loc=cfg["price_mu"], scale=cfg["price_sigma"], size=n).astype(float)
            sizes = np.abs(rng.normal(loc=cfg["size_mu"], scale=cfg["size_sigma"], size=n)).astype(float) + 1e-8

            # order ids and sequence ids
            order_ids = [int(seq.next()) for _ in range(n)]
            seq_ids = list(range(total, total + n))

            df = pd.DataFrame(
                {
                    "meta__timestamp": pd.to_datetime(timestamps, utc=True),
                    "meta__sequence_id": seq_ids,
                    "event_type": list(event_types),
                    "order_id": order_ids,
                    "side": list(sides),
                    "price": prices,
                    "size": sizes,
                    "exchange": ["EXCH"] * n,
                    "symbol": ["BTCUSD"] * n,
                }
            )

            # write as part file
            writer.write(df, append=True)

            total += n
            part_idx += 1
            elapsed = time.time() - started
            log.info(f"wrote part {part_idx} — total {total}/{rows_target} events — elapsed {elapsed:.1f}s")

        # merge parts into single parquet file
        writer.finalize()
        manifest = writer.get_manifest()
        return {"rows": total, "manifest": manifest, "timing": {"seconds": time.time() - started}}
    except Exception:
        log.exception("Engine raised exception:")
        # ensure finalize attempt to salvage parts
        try:
            writer.finalize()
        except Exception:
            log.exception("Finalize also failed.")
        raise


if __name__ == "__main__":
    log.info("Starting orderbook_l3 engine...")
    res = run_engine()
    log.info("Done.")
    print(res)
