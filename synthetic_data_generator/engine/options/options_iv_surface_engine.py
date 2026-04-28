# synthetic_data_generator/engine/options/options_iv_surface_engine.py
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    rel = cfg.get("paths", {}).get("options_iv_surface")
    if not rel:
        raise ValueError("paths.options_iv_surface must be set in configuration.")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "options_iv_surface.parquet"


def _build_expiry_schedule(n_expiries: int, start_ts: pd.Timestamp) -> List[pd.Timestamp]:
    # Spread expiries between 7 days and ~210 days to produce smooth term structure
    day_grid = np.geomspace(7, 210, num=n_expiries)
    expiries = [start_ts + pd.Timedelta(days=float(d)) for d in day_grid]
    return [ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC") for ts in expiries]


def _compute_surface_parameters(
    expiry_ts: pd.Timestamp, start_ts: pd.Timestamp, base_spot: float, level_shift: float
) -> Tuple[float, float]:
    days = max(1.0, (expiry_ts - start_ts).total_seconds() / 86_400)
    term_structure = 1.0 + 0.05 * np.log1p(days / 30.0) + level_shift
    spot_level = base_spot * (1.0 + 0.01 * np.log1p(days) + level_shift)
    return float(term_structure), float(spot_level)


def _design_strike_grid(n_strikes: int) -> np.ndarray:
    low, high = 0.75, 1.30
    return np.linspace(low, high, num=n_strikes)


def _calculate_volatility(
    moneyness: float,
    term_structure: float,
    expiry_days: int,
    option_type: str,
    skew_base: float,
    atm_vol_base: float,
) -> Tuple[float, float, float]:
    smile_component = 1.0 + 0.55 * (abs(np.log(max(moneyness, 1e-8)))) ** 1.25
    base_skew = skew_base * (1.0 + abs(moneyness - 1.0))
    if option_type == "P":
        skew = base_skew * 1.15
        asymmetry = 1.05 + 0.05 * abs(1.0 - moneyness)
    else:
        skew = base_skew * 0.8
        asymmetry = 0.98 + 0.03 * abs(1.0 - moneyness)
    term_factor = term_structure * (1.0 + 0.01 * np.sqrt(max(expiry_days, 1)) )
    implied_vol = atm_vol_base * smile_component * term_factor * (1.0 + skew) * asymmetry
    return float(implied_vol), float(skew), float(term_factor)


def _prepare_dimensions(total_rows: int) -> Tuple[int, List[int]]:
    n_expiries = max(4, min(10, total_rows // 500 + 4))
    strikes_per_expiry = max(6, min(24, total_rows // (2 * n_expiries)))
    strike_counts = [strikes_per_expiry for _ in range(n_expiries)]
    planned_rows = 2 * sum(strike_counts)

    idx = 0
    while planned_rows + 2 <= total_rows:
        strike_counts[idx % n_expiries] += 1
        planned_rows += 2
        idx += 1
    return n_expiries, strike_counts


def _validate_dataframe(df: pd.DataFrame, partition_date: str) -> None:
    if not df["meta__timestamp"].is_monotonic_increasing:
        raise ValueError("meta__timestamp must be monotonic increasing")
    if not df["meta__sequence_id"].is_monotonic_increasing:
        raise ValueError("meta__sequence_id must be strictly increasing")
    if (df["implied_vol"] <= 0).any() or df["implied_vol"].isna().any():
        raise ValueError("implied_vol must be positive and non-null")

    computed_days = ((df["expiry_ts"] - df["meta__timestamp"]).dt.total_seconds() / 86_400).round().astype(int)
    if not np.all(computed_days.values == df["expiry_days"].values):
        raise ValueError("expiry_days do not align with expiry_ts and meta__timestamp")

    if (df["moneyness"] <= 0).any() or df["moneyness"].isna().any():
        raise ValueError("moneyness must be positive and non-null")

    if not all(df["date"].astype(str) == partition_date):
        raise ValueError("date column must match partition date")


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
    """Generate synthetic options implied volatility surface data."""

    t0 = time.time()

    cfg = dict(config) if config else load_config()

    n_total = int(rows or cfg.get("rows", {}).get("options_iv_surface", 0))
    if n_total <= 0:
        raise ValueError("Row count for options IV surface engine must be positive.")

    global_cfg = cfg.get("global", {})
    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))

    chunk_size = _derive_chunk_size(cfg, n_total)
    output_path = _resolve_output_path(cfg, resolved_start)

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    base_spot = float(cfg.get("options_iv_surface", {}).get("base_spot", 200.0))
    base_atm_vol = float(cfg.get("options_iv_surface", {}).get("base_atm_vol", 0.22))
    skew_base = float(cfg.get("options_iv_surface", {}).get("skew_base", 0.04))

    n_expiries, strike_counts = _prepare_dimensions(n_total)
    expiries = _build_expiry_schedule(n_expiries, resolved_start)

    records: List[Dict[str, Any]] = []

    for exp_idx, expiry_ts in enumerate(expiries):
        strikes_here = strike_counts[exp_idx]
        term_structure, spot_level = _compute_surface_parameters(
            expiry_ts=expiry_ts, start_ts=resolved_start, base_spot=base_spot, level_shift=0.01 * exp_idx
        )
        strike_grid = _design_strike_grid(strikes_here)
        atm_vol_here = base_atm_vol * (1.0 + 0.02 * exp_idx)

        for rel_strike in strike_grid:
            strike = spot_level * float(rel_strike)
            moneyness = strike / spot_level
            expiry_days = int(round((expiry_ts - resolved_start).total_seconds() / 86_400))

            for option_type in ("C", "P"):
                if len(records) >= n_total:
                    break
                ts = clock.next()
                seq_id = seq.next()
                implied_vol, skew, term_factor = _calculate_volatility(
                    moneyness=moneyness,
                    term_structure=term_structure,
                    expiry_days=expiry_days,
                    option_type=option_type,
                    skew_base=skew_base,
                    atm_vol_base=atm_vol_here,
                )
                implied_vol = max(implied_vol, 0.01)

                records.append(
                    {
                        "meta__timestamp": ts,
                        "meta__sequence_id": seq_id,
                        "exchange": exchange,
                        "symbol": symbol,
                        "expiry_ts": expiry_ts,
                        "expiry_days": expiry_days,
                        "strike": float(strike),
                        "option_type": option_type,
                        "implied_vol": float(implied_vol),
                        "moneyness": float(moneyness),
                        "skew": float(skew),
                        "term_structure": float(term_factor),
                        "date": ts.strftime("%Y-%m-%d"),
                    }
                )
        if len(records) >= n_total:
            break

    if len(records) < n_total:
        # Pad with near-ATM call rows if odd counts remain
        while len(records) < n_total:
            ts = clock.next()
            seq_id = seq.next()
            expiry_ts = expiries[-1]
            expiry_days = int(round((expiry_ts - resolved_start).total_seconds() / 86_400))
            term_structure, spot_level = _compute_surface_parameters(
                expiry_ts=expiry_ts, start_ts=resolved_start, base_spot=base_spot, level_shift=0.02
            )
            moneyness = 1.0
            implied_vol, skew, term_factor = _calculate_volatility(
                moneyness=moneyness,
                term_structure=term_structure,
                expiry_days=expiry_days,
                option_type="C",
                skew_base=skew_base,
                atm_vol_base=base_atm_vol,
            )
            records.append(
                {
                    "meta__timestamp": ts,
                    "meta__sequence_id": seq_id,
                    "exchange": exchange,
                    "symbol": symbol,
                    "expiry_ts": expiry_ts,
                    "expiry_days": expiry_days,
                    "strike": float(spot_level),
                    "option_type": "C",
                    "implied_vol": float(max(implied_vol, 0.01)),
                    "moneyness": float(moneyness),
                    "skew": float(skew),
                    "term_structure": float(term_factor),
                    "date": ts.strftime("%Y-%m-%d"),
                }
            )

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.Categorical(df["date"], ordered=False)

    partition_date = resolved_start.strftime("%Y-%m-%d")
    _validate_dataframe(df, partition_date)

    writer = ParquetWriter(
        base_path=output_path, compression=cfg.get("writer", {}).get("compression", "snappy")
    )

    total_rows = len(df)
    if chunk_size >= total_rows:
        writer.write(df, append=False)
        print(f"[options_iv_surface] wrote {total_rows}/{total_rows} rows (single file) in {time.time() - t0:.2f}s")
    else:
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            writer.write(df.iloc[start_idx:end_idx])
            elapsed = time.time() - t0
            print(
                f"[options_iv_surface] wrote {end_idx}/{total_rows} rows (batch={end_idx - start_idx}) in {elapsed:.2f}s"
            )

    writer.finalize()
    manifest = writer.get_manifest()

    return {
        "rows": int(total_rows),
        "manifest": manifest,
        "timing": {"seconds": time.time() - t0},
    }


if __name__ == "__main__":
    import json

    result = run_engine()
    print(json.dumps(result, indent=2))
