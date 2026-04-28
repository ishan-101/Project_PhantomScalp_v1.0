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
_io_writer_mod._HAS_PYARROW = False


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
    rel = cfg.get("paths", {}).get("greeks_primary", "greeks/primary")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "greeks_primary.parquet"


def _generate_underlying_prices(rng: RNG, count: int, start_price: float, drift: float, step_vol: float) -> tuple[np.ndarray, float]:
    changes = rng.normal(loc=drift, scale=step_vol, size=count)
    changes = np.clip(changes, -0.03, 0.03)
    prices = np.empty(count, dtype=float)
    last_price = max(start_price, 1.0)
    for idx, delta in enumerate(changes):
        last_price = last_price * (1.0 + float(delta))
        last_price = max(last_price, 1.0)
        prices[idx] = last_price
    return prices, float(last_price)


def _generate_strikes(rng: RNG, underlying: np.ndarray) -> np.ndarray:
    skew = rng.normal(loc=0.0, scale=0.08, size=len(underlying))
    moneyness = 1.0 + skew
    strikes = underlying * moneyness
    strikes = np.clip(strikes, underlying * 0.6, underlying * 1.6)
    return strikes


def _generate_option_types(rng: RNG, count: int) -> np.ndarray:
    return rng.choice(["C", "P"], size=count, p=[0.55, 0.45])


def _generate_expiries(start_ts: pd.Timestamp, rng: RNG, count: int) -> np.ndarray:
    base_date = start_ts.tz_convert("UTC").normalize()
    days = rng.integers(low=7, high=120, size=count)
    expiries = base_date + pd.to_timedelta(days, unit="D")
    return pd.to_datetime(expiries).tz_convert("UTC")


def _compute_greeks(option_types: np.ndarray, underlying: np.ndarray, strikes: np.ndarray, expiries: pd.DatetimeIndex, ivs: np.ndarray) -> Dict[str, np.ndarray]:
    moneyness = underlying / strikes
    now_utc = pd.Timestamp.now(tz="UTC")
    time_to_expiry = np.maximum((expiries - now_utc).days, 7)
    time_factor = np.clip(time_to_expiry / 365.0, 0.05, 1.5)

    call_mask = option_types == "C"
    put_mask = ~call_mask

    delta = np.empty_like(moneyness, dtype=float)
    delta[call_mask] = np.clip(0.5 + 0.6 * (moneyness[call_mask] - 1.0), 0.01, 0.99)
    delta[put_mask] = -np.clip(0.5 + 0.6 * (1.0 - moneyness[put_mask]), 0.01, 0.99)

    gamma_base = np.exp(-np.abs(moneyness - 1.0) * 4.0) * 0.05
    gamma = np.clip(gamma_base * (ivs + 0.1), 0.0005, 0.5)

    theta = -(0.02 + 0.2 * (1.0 / np.sqrt(time_factor))) * (ivs + 0.1)
    theta = np.clip(theta, -5.0, -0.0001)

    vega = np.clip(0.3 * np.sqrt(time_factor) * (1.0 + np.abs(moneyness - 1.0)), 0.01, 3.5)

    rho = np.empty_like(moneyness, dtype=float)
    rho[call_mask] = np.clip(0.02 * time_factor[call_mask], 0.001, 0.5)
    rho[put_mask] = -np.clip(0.02 * time_factor[put_mask], 0.001, 0.5)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# Public engine API
# ---------------------------------------------------------------------------

def run_engine(
    config: Optional[Dict[str, Any]] = None,
    rows: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
    exchange: str = "EX",
    symbol: str = "BTC-OPT",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.time()

    cfg = dict(config) if config else load_config()

    global_cfg = cfg.get("global", {})
    resolved_seed = seed if seed is not None else global_cfg.get("seed")

    n_total = int(rows or cfg.get("rows", {}).get("greeks_primary", 0))
    if n_total <= 0:
        raise ValueError("Row count for greeks primary engine must be positive.")

    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))

    chunk_size = _derive_chunk_size(cfg, n_total)

    output_path = _resolve_output_path(cfg, resolved_start)

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(resolved_seed)
    rng = RNG(seed=resolved_seed)

    writer = ParquetWriter(base_path=output_path, compression=cfg.get("writer", {}).get("compression", "snappy"))

    remaining = n_total
    produced = 0
    price_cursor = float(cfg.get("greeks_primary", {}).get("start_price", 30_000.0))
    drift = float(cfg.get("greeks_primary", {}).get("drift", 0.00002))
    step_vol = float(cfg.get("greeks_primary", {}).get("step_volatility", 0.001))

    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        timestamps = [clock.next() for _ in range(batch_size)]
        timestamp_index = pd.DatetimeIndex(timestamps)

        underlying_prices, price_cursor = _generate_underlying_prices(rng, batch_size, price_cursor, drift, step_vol)
        strikes = _generate_strikes(rng, underlying_prices)
        option_types = _generate_option_types(rng, batch_size)
        expiries = _generate_expiries(resolved_start, rng, batch_size)
        implied_vols = np.clip(rng.normal(loc=0.65, scale=0.15, size=batch_size), 0.05, 2.5)
        greeks = _compute_greeks(option_types, underlying_prices, strikes, expiries, implied_vols)

        sequence_ids = seq.next_batch(batch_size)

        df = pd.DataFrame(
            {
                "meta__timestamp": timestamp_index,
                "meta__sequence_id": sequence_ids,
                "exchange": exchange,
                "symbol": symbol,
                "option_type": option_types,
                "strike": strikes,
                "expiry": expiries,
                "underlying_price": underlying_prices,
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta": greeks["theta"],
                "vega": greeks["vega"],
                "rho": greeks["rho"],
                "implied_volatility": implied_vols,
            }
        )
        df["date"] = df["meta__timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")
        writer.write(df)

        produced += batch_size
        remaining -= batch_size
        elapsed = time.time() - t0
        print(f"[greeks_primary] wrote {produced}/{n_total} rows (batch={batch_size}) in {elapsed:.2f}s")

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
