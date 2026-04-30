"""
Cross-asset correlation engine.
Generates smooth, regime-aware synthetic correlations between asset pairs.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    rel = cfg.get("paths", {}).get("crossasset_corr", "crossasset/correlation")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "crossasset_correlation.parquet"


def _generate_universe(cfg: Dict[str, Any], rng: RNG) -> Tuple[List[str], List[str]]:
    params = cfg.get("crossasset_correlation", {})
    assets: List[str]
    exchanges: List[str]

    if isinstance(params.get("assets"), Sequence) and params.get("assets"):
        assets = [str(a) for a in params.get("assets", [])]
    else:
        count = int(params.get("asset_count", 6))
        assets = [f"AST{idx:02d}" for idx in range(1, count + 1)]

    if isinstance(params.get("exchanges"), Sequence) and params.get("exchanges"):
        exchanges = [str(ex) for ex in params.get("exchanges", [])]
    else:
        exch_count = int(params.get("exchange_count", 2))
        exchanges = [f"EXCH{idx:02d}" for idx in range(1, exch_count + 1)]

    assets = [assets[i] for i in np.argsort(rng.integers(low=0, high=10_000, size=len(assets)))]

    return assets, exchanges


def _pair_cycle(assets: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i, ax in enumerate(assets):
        for ay in assets[i + 1 :]:
            pairs.append((ax, ay))
    if not pairs:
        pairs = [(assets[0], assets[0])] if assets else [("AST00", "AST00")]
    return pairs


def _smooth_step(prev: float, target: float, noise: float) -> float:
    drift = (target - prev) * 0.05
    stepped = prev + drift + noise
    return float(np.clip(stepped, -0.98, 0.98))


def _regime_center(regime: str) -> float:
    if regime == "high":
        return 0.75
    if regime == "low":
        return 0.1
    return 0.4


def _update_regime(rng: RNG, current: str) -> str:
    transition = rng.choice(["stay", "flip_low", "flip_high"], p=[0.85, 0.075, 0.075])
    if transition == "flip_low":
        return "low"
    if transition == "flip_high":
        return "high"
    return current


def _compute_metrics(base_corr: float, rng: RNG) -> Dict[str, float]:
    jitter_short = float(rng.normal(loc=0.0, scale=0.03))
    jitter_medium = float(rng.normal(loc=0.0, scale=0.02))
    jitter_long = float(rng.normal(loc=0.0, scale=0.01))

    returns_corr_1m = float(np.clip(base_corr + jitter_short, -1.0, 1.0))
    returns_corr_5m = float(np.clip(base_corr * 0.98 + jitter_short * 0.7, -1.0, 1.0))
    returns_corr_15m = float(np.clip(base_corr * 0.97 + jitter_medium, -1.0, 1.0))
    returns_corr_1h = float(np.clip(base_corr * 0.95 + jitter_medium * 0.7, -1.0, 1.0))
    returns_corr_4h = float(np.clip(base_corr * 0.93 + jitter_long, -1.0, 1.0))
    returns_corr_1d = float(np.clip(base_corr * 0.9 + jitter_long * 0.5, -1.0, 1.0))

    vol_x = float(np.clip(abs(rng.normal(loc=0.6, scale=0.15)), 0.05, None))
    vol_y = float(np.clip(abs(rng.normal(loc=0.5, scale=0.12)), 0.05, None))
    covariance = float(base_corr * vol_x * vol_y * 0.01)
    beta_xy = float(covariance / max(vol_y ** 2, 1e-6))
    volatility_ratio = float(vol_x / max(vol_y, 1e-6))

    lead_lag_wave = float(np.tanh(base_corr) * 0.5 + float(rng.normal(loc=0.0, scale=0.02)))
    correlation_z = (returns_corr_15m - _regime_center("neutral")) / 0.1
    correlation_regime = "neutral"
    if correlation_z > 0.5:
        correlation_regime = "high"
    elif correlation_z < -0.5:
        correlation_regime = "low"

    return {
        "returns_corr_1m": returns_corr_1m,
        "returns_corr_5m": returns_corr_5m,
        "returns_corr_15m": returns_corr_15m,
        "returns_corr_1h": returns_corr_1h,
        "returns_corr_4h": returns_corr_4h,
        "returns_corr_1d": returns_corr_1d,
        "rolling_covariance": covariance,
        "beta_xy": beta_xy,
        "volatility_ratio": volatility_ratio,
        "lead_lag_score": lead_lag_wave,
        "correlation_zscore": float(correlation_z),
        "correlation_regime": correlation_regime,
    }


# ---------------------------------------------------------------------------
# Public engine API
# ---------------------------------------------------------------------------

def run_engine(
    config: Optional[Dict[str, Any]] = None,
    rows: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Generate synthetic cross-asset correlations.

    Args:
        config: Optional preloaded configuration dictionary.
        rows: Optional override for the number of rows; defaults to rows.crossasset_correlation.
        start_ts: Optional UTC-coercible start timestamp.
    """

    t0 = time.time()
    cfg = dict(config) if config else load_config()
    global_cfg = cfg.get("global", {})

    total_rows = int(rows or cfg.get("rows", {}).get("crossasset_correlation", 0))
    if total_rows <= 0:
        raise ValueError("Row count for crossasset_correlation must be positive.")

    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"))
    output_path = _resolve_output_path(cfg, resolved_start)
    chunk_size = _derive_chunk_size(cfg, total_rows)

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    assets, exchanges = _generate_universe(cfg, rng)
    pairs = _pair_cycle(assets)

    regimes = {pair: rng.choice(["neutral", "high", "low"], p=[0.7, 0.15, 0.15]) for pair in pairs}
    base_levels = {pair: _regime_center(regimes[pair]) for pair in pairs}

    writer = ParquetWriter(base_path=output_path, compression=cfg.get("writer", {}).get("compression", "snappy"))
    single_file_mode = total_rows <= chunk_size

    remaining = total_rows
    produced = 0
    parts_written = 0
    records: List[Dict[str, Any]] = []

    wrote_canonical = False

    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        records.clear()

        for _ in range(batch_size):
            pair = pairs[produced % len(pairs)]
            regime = regimes[pair]
            target = _regime_center(regime)
            noise = float(rng.normal(loc=0.0, scale=0.015))
            base_levels[pair] = _smooth_step(base_levels[pair], target, noise)

            if rng.choice([True, False], p=[0.02, 0.98]):
                regimes[pair] = _update_regime(rng, regime)

            metrics = _compute_metrics(base_levels[pair], rng)

            ts = clock.next()
            records.append(
                {
                    "meta__timestamp": ts,
                    "meta__sequence_id": seq.next(),
                    "date": ts.date().isoformat(),
                    "asset_x": pair[0],
                    "asset_y": pair[1],
                    "exchange": str(rng.choice(exchanges)),
                    **metrics,
                }
            )
            produced += 1

        df = pd.DataFrame.from_records(records)

        # Ensure stable (non-dictionary) string types for Arrow schema merging across part files
        string_cols = ["date", "asset_x", "asset_y", "exchange", "correlation_regime"]
        existing_string_cols = [col for col in string_cols if col in df.columns]
        if existing_string_cols:
            df[existing_string_cols] = df[existing_string_cols].astype(str)
        if single_file_mode:
            writer.write(df, append=False)
            wrote_canonical = True
            remaining = 0
        else:
            writer.write(df)
            parts_written += 1
            remaining -= batch_size

        elapsed = time.time() - t0
        print(f"[crossasset_correlation] wrote {produced}/{total_rows} rows (batch={batch_size}) in {elapsed:.2f}s")

    if not wrote_canonical:
        writer.finalize()
    manifest = writer.get_manifest()

    return {
        "rows": total_rows,
        "manifest": manifest,
        "timing": {"seconds": time.time() - t0, "parts": parts_written},
    }


if __name__ == "__main__":
    import json

    result = run_engine()
    print(json.dumps(result, indent=2))
