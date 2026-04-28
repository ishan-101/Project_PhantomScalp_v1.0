# synthetic_data_generator/engine/spot/spot_engine.py
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Import helpers (no try/except around imports)
# ---------------------------------------------------------------------------

def _add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _import_module_if_available(module_path: str):
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        return None
    return importlib.import_module(module_path)


_add_repo_root_to_path()

_loader_mod = (
    _import_module_if_available("synthetic_data_generator.engine.config.loader")
    or _import_module_if_available("engine.config.loader")
)
if _loader_mod is None:
    raise ImportError("Unable to locate config loader module.")

_utils_base = None
for candidate in ("synthetic_data_generator.engine.utils", "engine.utils"):
    if importlib.util.find_spec(candidate) is not None:
        _utils_base = candidate
        break
if _utils_base is None:
    raise ImportError("Unable to locate utils package for engine dependencies.")

_clock_mod = importlib.import_module(f"{_utils_base}.clock")
_sequence_mod = importlib.import_module(f"{_utils_base}.sequence_id")
_rng_mod = importlib.import_module(f"{_utils_base}.rng")
_io_writer_mod = importlib.import_module(f"{_utils_base}.io_writer")

# Exported utilities
load_config = getattr(_loader_mod, "load_config")
CanonicalClock = getattr(_clock_mod, "CanonicalClock")
SequenceID = getattr(_sequence_mod, "SequenceID")
RNG = getattr(_rng_mod, "RNG")
ParquetWriter = getattr(_io_writer_mod, "ParquetWriter")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _coerce_utc(ts: Optional[pd.Timestamp], default: Optional[str] = None) -> pd.Timestamp:
    if ts is not None:
        out = pd.to_datetime(ts)
    elif default is not None:
        out = pd.to_datetime(default)
    else:
        out = pd.Timestamp.utcnow()
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _derive_chunk_size(cfg: Dict[str, Any], total_rows: int) -> int:
    sharding = cfg.get("partitioning", {}).get("sharding", {})
    max_rows = sharding.get("max_rows_per_file")
    min_rows = sharding.get("min_rows_per_file")
    if max_rows is None and min_rows is None:
        return total_rows
    if max_rows is None:
        return max(1, min(total_rows, int(min_rows)))
    return max(1, min(total_rows, int(max_rows)))


def _resolve_output_path(cfg: Dict[str, Any], partition_date: pd.Timestamp) -> Path:
    base = cfg.get("paths", {}).get("base")
    if not base:
        raise ValueError("paths.base must be set in configuration.")
    rel = cfg.get("paths", {}).get("spot", "spot")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "spot.parquet"


def _generate_price_series(rng: RNG, count: int, start_price: float, drift: float, step_vol: float) -> tuple[np.ndarray, float]:
    changes = rng.normal(loc=drift, scale=step_vol, size=count)
    changes = np.clip(changes, -0.02, 0.02)
    prices = np.empty(count, dtype=float)
    last_price = max(start_price, 1e-6)
    for idx, delta in enumerate(changes):
        last_price = last_price * (1.0 + float(delta))
        if last_price <= 0:
            last_price = max(start_price * 0.5, 1e-6)
        prices[idx] = last_price
    return prices, float(last_price)


def _generate_sizes(rng: RNG, count: int) -> np.ndarray:
    sizes = np.exp(rng.normal(loc=-2.0, scale=0.9, size=count))
    return np.clip(sizes, 1e-6, None)


def _generate_sides(rng: RNG, count: int) -> np.ndarray:
    return rng.choice(["B", "S"], size=count)


# ---------------------------------------------------------------------------
# Public engine API
# ---------------------------------------------------------------------------

def run_engine(
    config: Optional[Dict[str, Any]] = None,
    rows: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
    exchange: str = "EX",
    symbol: str = "SYM",
) -> Dict[str, Any]:
    """Generate synthetic spot trade data.

    Args:
        config: Optional preloaded configuration dictionary; if omitted the
            centralized configuration loader is used.
        rows: Optional override for the number of rows; defaults to
            ``rows.spot`` from config.
        start_ts: Optional UTC-coercible timestamp for the first event.
        exchange: Exchange identifier written to every row.
        symbol: Symbol identifier written to every row.

    Returns:
        Mapping containing total rows emitted, writer manifest (path/rows/bytes)
        and basic timing information.
    """
    t0 = time.time()

    cfg = dict(config) if config else load_config()

    n_total = int(rows or cfg.get("rows", {}).get("spot", 0))
    if n_total <= 0:
        raise ValueError("Row count for spot engine must be positive.")

    global_cfg = cfg.get("global", {})
    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))

    chunk_size = _derive_chunk_size(cfg, n_total)

    output_path = _resolve_output_path(cfg, resolved_start)

    clock_cfg = {"start_ts": resolved_start.isoformat(), "inter_event_us": global_cfg.get("inter_event_us", 1000)}
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    writer = ParquetWriter(base_path=output_path, compression=cfg.get("writer", {}).get("compression", "snappy"))

    remaining = n_total
    produced = 0
    price_cursor = float(cfg.get("spot", {}).get("start_price", 30_000.0))
    drift = float(cfg.get("spot", {}).get("drift", 0.00001))
    step_vol = float(cfg.get("spot", {}).get("step_volatility", 0.0015))

    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        timestamps = [clock.next() for _ in range(batch_size)]
        price_series, price_cursor = _generate_price_series(rng, batch_size, price_cursor, drift, step_vol)
        sizes = _generate_sizes(rng, batch_size)
        sides = _generate_sides(rng, batch_size)
        sequence_ids = seq.next_batch(batch_size)

        df = pd.DataFrame(
            {
                "meta__timestamp": timestamps,
                "meta__sequence_id": sequence_ids,
                "exchange": exchange,
                "symbol": symbol,
                "price": price_series,
                "size": sizes,
                "side": sides,
            }
        )

        writer.write(df)

        produced += batch_size
        remaining -= batch_size
        elapsed = time.time() - t0
        print(f"[spot] wrote {produced}/{n_total} rows (batch={batch_size}) in {elapsed:.2f}s")

    writer.finalize()
    manifest = writer.get_manifest()

    return {
        "rows": n_total,
        "manifest": manifest,
        "timing": {"seconds": time.time() - t0},
    }


if __name__ == "__main__":
    import json

    result = run_engine()
    print(json.dumps(result, indent=2))
