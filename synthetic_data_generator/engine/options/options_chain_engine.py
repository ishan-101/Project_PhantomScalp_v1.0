# synthetic_data_generator/engine/options/options_chain_engine.py
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
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


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
    rel = cfg.get("paths", {}).get("options_chain")
    if not rel:
        raise ValueError("paths.options_chain must be set in configuration.")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "options_chain.parquet"


def _generate_quotes(
    rng: RNG,
    count: int,
    spot_anchor: float,
    expiry_days: np.ndarray,
) -> Dict[str, np.ndarray]:
    moneyness = rng.normal(loc=1.0, scale=0.15, size=count)
    strikes = spot_anchor * moneyness
    strikes = np.clip(strikes, spot_anchor * 0.35, spot_anchor * 1.8)

    option_types = rng.choice(["C", "P"], size=count)

    intrinsic_value = np.maximum(0.0, (spot_anchor - strikes) * (option_types == "P"))
    intrinsic_value += np.maximum(0.0, (strikes - spot_anchor) * (option_types == "C"))

    time_value_scale = 1.0 + (expiry_days.astype(float) / 180.0)
    base_vol = 0.22 + 0.05 * np.abs(np.log(np.maximum(strikes / spot_anchor, 1e-6)))
    premium = np.maximum(0.5, base_vol * spot_anchor * 0.015 * time_value_scale)
    premium = premium + intrinsic_value

    bid_spread = rng.uniform(0.005, 0.018, size=count)
    ask_spread = rng.uniform(0.006, 0.02, size=count)

    mid = premium
    bid = np.maximum(0.01, mid * (1.0 - bid_spread))
    ask = mid * (1.0 + ask_spread)

    last = mid + rng.normal(loc=0.0, scale=mid * 0.02)
    last = np.maximum(0.01, last)

    open_interest = rng.integers(low=50, high=4000, size=count)
    volume = rng.integers(low=1, high=np.maximum(open_interest // 10, 5) + 5)

    return {
        "strike": strikes.astype(float),
        "option_type": option_types,
        "bid": bid.astype(float),
        "ask": ask.astype(float),
        "mid": mid.astype(float),
        "last": last.astype(float),
        "volume": volume.astype(int),
        "open_interest": open_interest.astype(int),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_engine(
    config: Optional[Dict[str, Any]] = None,
    rows: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
    exchange: str = "EX",
    symbol: str = "SYM",
) -> Dict[str, Any]:
    t0 = time.time()

    cfg = dict(config) if config else load_config()

    total_rows = int(rows or cfg.get("rows", {}).get("options_chain", 0))
    if total_rows <= 0:
        raise ValueError("Row count for options chain engine must be positive.")

    global_cfg = cfg.get("global", {})
    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))
    partition_date = resolved_start.tz_convert("UTC").normalize()

    chunk_size = _derive_chunk_size(cfg, total_rows)
    output_path = _resolve_output_path(cfg, partition_date)
    staging_parent = output_path.parent.parent / "_temp_options_chain"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_path = staging_parent / output_path.name

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    expiry_candidates = np.array([7, 14, 30, 45, 60, 90, 120, 180])
    expiry_probs = np.array([0.12, 0.14, 0.16, 0.14, 0.14, 0.12, 0.1, 0.08])

    spot_anchor = float(cfg.get("options", {}).get("spot_anchor", 30000.0))

    writer = ParquetWriter(base_path=staging_path, compression=cfg.get("writer", {}).get("compression", "snappy"))

    records = []
    for _ in range(total_rows):
        ts = clock.next()
        seq_id = seq.next()

        expiry_days = int(rng.choice(expiry_candidates, p=expiry_probs))
        expiry_ts = ts + pd.Timedelta(days=expiry_days)

        quote_metrics = _generate_quotes(rng, count=1, spot_anchor=spot_anchor, expiry_days=np.array([expiry_days]))

        records.append(
            {
                "meta__timestamp": ts,
                "meta__sequence_id": seq_id,
                "exchange": exchange,
                "symbol": symbol,
                "expiry_ts": expiry_ts,
                "expiry_days": expiry_days,
                "strike": float(quote_metrics["strike"][0]),
                "option_type": quote_metrics["option_type"][0],
                "bid": float(quote_metrics["bid"][0]),
                "ask": float(quote_metrics["ask"][0]),
                "mid": float(quote_metrics["mid"][0]),
                "last": float(quote_metrics["last"][0]),
                "volume": int(quote_metrics["volume"][0]),
                "open_interest": int(quote_metrics["open_interest"][0]),
                "date": ts.strftime("%Y-%m-%d"),
            }
        )

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.Categorical(df["date"], ordered=False)

    if not df["meta__timestamp"].is_monotonic_increasing:
        raise ValueError("meta__timestamp must be monotonic increasing")
    if not df["meta__sequence_id"].is_monotonic_increasing:
        raise ValueError("meta__sequence_id must be strictly increasing")

    total_rows = len(df)
    if chunk_size >= total_rows:
        writer.write(df, append=False)
        print(f"[options_chain] wrote {total_rows}/{total_rows} rows (single file) in {time.time() - t0:.2f}s")
    else:
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            writer.write(df.iloc[start_idx:end_idx])
            elapsed = time.time() - t0
            print(
                f"[options_chain] wrote {end_idx}/{total_rows} rows (batch={end_idx - start_idx}) in {elapsed:.2f}s"
            )

    writer.finalize()

    if staging_path.exists():
        staging_path.replace(output_path)

    try:
        if staging_parent.exists() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()
    except Exception:
        pass

    manifest = writer.get_manifest()
    manifest["path"] = str(output_path)

    return {
        "rows": int(total_rows),
        "manifest": manifest,
        "timing": {"seconds": time.time() - t0},
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    import json

    result = run_engine()
    print(json.dumps(result, indent=2))