"""
greeks_flow_engine.py

Deterministic synthetic Greeks flow generator.

Schema (column order):
1. meta__timestamp (tz-aware UTC)
2. meta__sequence_id (int)
3. exchange (string)
4. symbol (string)
5. date (YYYY-MM-DD string)
6. option_type (call/put)
7. strike (float64)
8. expiry (YYYY-MM-DD string)
9. delta (float64)
10. gamma (float64)
11. theta (float64)
12. vega (float64)
13. rho (float64)
14. delta_flow (float64)
15. gamma_flow (float64)
16. vega_flow (float64)
17. oi_change (float64)
18. iv_change (float64)
19. price_change (float64)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
_schema_validator_mod = importlib.import_module(f"{_utils_base}.schema_validator")

# Exported utilities
load_config = getattr(_loader_mod, "load_config")
CanonicalClock = getattr(_clock_mod, "CanonicalClock")
SequenceID = getattr(_sequence_mod, "SequenceID")
RNG = getattr(_rng_mod, "RNG")
ParquetWriter = getattr(_io_writer_mod, "ParquetWriter")
validate_basic_tick_schema = getattr(_schema_validator_mod, "validate_basic_tick_schema")

# Force pandas merge path during finalize to avoid pyarrow dictionary merge conflicts
setattr(_io_writer_mod, "_HAS_PYARROW", False)


# ---------------------------------------------------------------------------
# Helpers
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
    rel = cfg.get("paths", {}).get("greeks_flow", "greeks/flow")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "greeks_flow.parquet"


def _init_option_universe() -> Tuple[list[str], list[str], list[str], dict[str, float]]:
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
    exchanges = ["OPRA", "GLOBEX"]
    expiries = [
        "2025-12-05",
        "2025-12-12",
        "2026-01-17",
        "2026-03-20",
    ]
    base_prices = {
        "SPY": 480.0,
        "QQQ": 400.0,
        "IWM": 200.0,
        "AAPL": 210.0,
        "MSFT": 430.0,
    }
    return symbols, exchanges, expiries, base_prices


def _bounded_update(value: float, delta: float, lower: float, upper: float) -> float:
    return float(np.clip(value + delta, lower, upper))


def _initialize_greek_levels(rng: RNG, base_price: float, moneyness: float) -> Dict[str, float]:
    # deterministic initial greeks anchored to moneyness and price scale
    delta = float(np.clip(0.5 + 0.4 * moneyness, -1.0, 1.0))
    gamma = float(np.clip(0.01 * (1.0 - abs(moneyness)), -0.05, 0.05))
    theta = float(np.clip(-0.02 * (1.0 + abs(moneyness)), -1.0, 0.0))
    vega = float(np.clip(0.1 * base_price / 100.0, 0.01, 5.0))
    rho = float(np.clip(0.5 * moneyness, -1.0, 1.0))
    # add deterministic gentle offsets
    delta += float(rng.normal(loc=0.0, scale=0.02))
    gamma += float(rng.normal(loc=0.0, scale=0.001))
    theta += float(rng.normal(loc=0.0, scale=0.002))
    vega += float(rng.normal(loc=0.0, scale=0.01))
    rho += float(rng.normal(loc=0.0, scale=0.01))
    return {
        "delta": float(np.clip(delta, -1.0, 1.0)),
        "gamma": float(np.clip(gamma, -0.1, 0.1)),
        "theta": float(np.clip(theta, -2.0, 0.0)),
        "vega": float(np.clip(vega, 0.0, 10.0)),
        "rho": float(np.clip(rho, -1.0, 1.0)),
    }


def _update_levels_and_flows(rng: RNG, levels: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    # small deterministic perturbations to represent flow (change over time)
    drift = {
        "delta": rng.normal(loc=0.0, scale=0.01),
        "gamma": rng.normal(loc=0.0, scale=0.0005),
        "theta": rng.normal(loc=0.0, scale=0.002),
        "vega": rng.normal(loc=0.0, scale=0.02),
        "rho": rng.normal(loc=0.0, scale=0.005),
    }

    new_levels = {
        "delta": _bounded_update(levels["delta"], drift["delta"], -1.0, 1.0),
        "gamma": _bounded_update(levels["gamma"], drift["gamma"], -0.1, 0.1),
        "theta": _bounded_update(levels["theta"], drift["theta"], -2.0, 0.0),
        "vega": _bounded_update(levels["vega"], drift["vega"], 0.0, 10.0),
        "rho": _bounded_update(levels["rho"], drift["rho"], -1.0, 1.0),
    }

    flows = {
        "delta_flow": float(new_levels["delta"] - levels["delta"]),
        "gamma_flow": float(new_levels["gamma"] - levels["gamma"]),
        "vega_flow": float(new_levels["vega"] - levels["vega"]),
    }
    return new_levels, flows


def _compute_price_and_changes(rng: RNG, current_price: float) -> Tuple[float, float]:
    pct_move = float(np.clip(rng.normal(loc=0.0005, scale=0.003), -0.02, 0.02))
    new_price = max(current_price * (1.0 + pct_move), 0.01)
    return new_price, float(new_price - current_price)


def _compute_flow_extras(rng: RNG) -> Tuple[float, float]:
    oi_change = float(np.clip(rng.normal(loc=0.0, scale=25.0), -150.0, 150.0))
    iv_change = float(np.clip(rng.normal(loc=0.0, scale=0.02), -0.1, 0.1))
    return oi_change, iv_change


# ---------------------------------------------------------------------------
# Public engine API
# ---------------------------------------------------------------------------


def run_engine(
    config: Optional[Dict[str, Any]] = None,
    rows: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Generate synthetic Greeks flow data."""

    t0 = time.time()

    cfg = dict(config) if config else load_config()

    n_total = int(rows or cfg.get("rows", {}).get("greeks_flow", 0))
    if n_total <= 0:
        raise ValueError("Row count for greeks_flow engine must be positive.")

    global_cfg = cfg.get("global", {})
    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))

    chunk_size = _derive_chunk_size(cfg, n_total)

    output_path = _resolve_output_path(cfg, resolved_start)

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
        "jitter_us": global_cfg.get("jitter_us", 0),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    writer = ParquetWriter(base_path=output_path, compression=cfg.get("writer", {}).get("compression", "snappy"))

    symbols, exchanges, expiries, base_prices = _init_option_universe()
    option_states: Dict[Tuple[str, str, float, str], Dict[str, Any]] = {}

    remaining = n_total
    produced = 0
    while remaining > 0:
        batch_size = min(chunk_size, remaining)

        batch_data = {"meta__timestamp": [], "meta__sequence_id": [], "exchange": [], "symbol": [], "date": [],
                      "option_type": [], "strike": [], "expiry": [], "delta": [], "gamma": [], "theta": [],
                      "vega": [], "rho": [], "delta_flow": [], "gamma_flow": [], "vega_flow": [],
                      "oi_change": [], "iv_change": [], "price_change": []}

        timestamps = [clock.next() for _ in range(batch_size)]
        sequence_ids = seq.next_batch(batch_size)

        for ts_val, seq_id in zip(timestamps, sequence_ids):
            symbol = rng.choice(symbols)
            exchange = rng.choice(exchanges)
            option_type = rng.choice(["call", "put"])

            base_price = base_prices[symbol]
            moneyness_shift = float(np.clip(rng.normal(loc=0.0, scale=0.1), -0.5, 0.5))
            strike = float(base_price * (1.0 + moneyness_shift))
            expiry = str(rng.choice(expiries))

            option_key = (symbol, option_type, strike, expiry)
            if option_key not in option_states:
                levels = _initialize_greek_levels(rng, base_price, moneyness_shift)
                price_level = base_price
                option_states[option_key] = {"levels": levels, "price": price_level}

            current_state = option_states[option_key]
            levels = current_state["levels"]
            price_level = current_state["price"]

            new_price, price_change = _compute_price_and_changes(rng, price_level)
            new_levels, flows = _update_levels_and_flows(rng, levels)
            oi_change, iv_change = _compute_flow_extras(rng)

            # Update state for next time this option is encountered
            option_states[option_key] = {"levels": new_levels, "price": new_price}

            batch_data["meta__timestamp"].append(ts_val)
            batch_data["meta__sequence_id"].append(seq_id)
            batch_data["exchange"].append(exchange)
            batch_data["symbol"].append(symbol)
            batch_data["date"].append(ts_val.strftime("%Y-%m-%d"))
            batch_data["option_type"].append(option_type)
            batch_data["strike"].append(float(strike))
            batch_data["expiry"].append(expiry)
            batch_data["delta"].append(new_levels["delta"])
            batch_data["gamma"].append(new_levels["gamma"])
            batch_data["theta"].append(new_levels["theta"])
            batch_data["vega"].append(new_levels["vega"])
            batch_data["rho"].append(new_levels["rho"])
            batch_data["delta_flow"].append(flows["delta_flow"])
            batch_data["gamma_flow"].append(flows["gamma_flow"])
            batch_data["vega_flow"].append(flows["vega_flow"])
            batch_data["oi_change"].append(oi_change)
            batch_data["iv_change"].append(iv_change)
            batch_data["price_change"].append(price_change)

        df = pd.DataFrame(batch_data)
        # enforce column order and dtypes
        column_order = [
            "meta__timestamp",
            "meta__sequence_id",
            "exchange",
            "symbol",
            "date",
            "option_type",
            "strike",
            "expiry",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "delta_flow",
            "gamma_flow",
            "vega_flow",
            "oi_change",
            "iv_change",
            "price_change",
        ]
        df = df[column_order]

        numeric_cols = [
            "strike",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "delta_flow",
            "gamma_flow",
            "vega_flow",
            "oi_change",
            "iv_change",
            "price_change",
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df["meta__sequence_id"] = df["meta__sequence_id"].astype(int)
        string_cols = ["exchange", "symbol", "date", "option_type", "expiry"]
        for col in string_cols:
            df[col] = pd.Series(df[col], dtype="string[pyarrow]")

        validate_basic_tick_schema(df, ["exchange", "symbol", "date"])

        writer.write(df)

        produced += batch_size
        remaining -= batch_size

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
