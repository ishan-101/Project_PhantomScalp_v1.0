# synthetic_data_generator/engine/options/options_oi_engine.py
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

def _coerce_utc(
    ts: Optional[pd.Timestamp],
    default: Optional[str] = None,
    tz_name: str = "UTC",
) -> pd.Timestamp:
    if ts is not None:
        out = pd.to_datetime(ts)
    elif default is not None:
        out = pd.to_datetime(default)
    else:
        out = pd.Timestamp.utcnow()
    if out.tzinfo is None:
        out = out.tz_localize(tz_name)
    else:
        out = out.tz_convert(tz_name)
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
    rel = cfg.get("paths", {}).get("options_oi", "options/oi")
    dated_dir = Path(base) / rel / f"date={partition_date.strftime('%Y-%m-%d')}"
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / "options_oi.parquet"


def _generate_option_metrics(
    rng: RNG,
    count: int,
    spot_anchor: float,
    expiry_days: np.ndarray,
) -> Dict[str, np.ndarray]:
    # strikes centered around spot_anchor with moderate dispersion
    moneyness = rng.normal(loc=0.0, scale=0.18, size=count)
    strikes = spot_anchor * (1.0 + moneyness * 0.2)
    strikes = np.clip(strikes, spot_anchor * 0.4, spot_anchor * 1.6)

    option_types = rng.choice(["C", "P"], size=count)

    distance = np.abs((strikes - spot_anchor) / spot_anchor)
    distance_factor = np.exp(-distance * 4.0)

    term_factor = 1.0 + (expiry_days.astype(float) / 90.0)

    stability_noise = np.clip(rng.normal(loc=1.0, scale=0.08, size=count), 0.8, 1.2)
    base_oi = 1200.0
    open_interest = base_oi * distance_factor * term_factor * stability_noise
    open_interest = np.maximum(open_interest, 0.0)
    open_interest = np.rint(open_interest).astype(int)

    churn_intensity = 0.04 + 0.18 * np.exp(-expiry_days / 25.0)
    max_change = np.maximum(1, np.rint(open_interest * churn_intensity)).astype(int)
    max_change = np.clip(max_change, 1, np.maximum(open_interest, 1))

    signed = rng.choice([-1, 1], size=count)
    oi_change = (signed * rng.integers(low=0, high=max_change + 1)).astype(int)

    oi_change = np.clip(
        oi_change,
        -np.maximum(open_interest, 0),
        np.maximum(open_interest, 0),
    )

    additional_flow = rng.integers(low=0, high=np.maximum(5, (open_interest * 0.12).astype(int)) + 1)
    volume = np.maximum(np.abs(oi_change), additional_flow)

    return {
        "strike": strikes.astype(float),
        "option_type": option_types,
        "open_interest": open_interest,
        "oi_change": oi_change,
        "volume": volume.astype(int),
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

    total_rows = int(rows or cfg.get("rows", {}).get("options_oi", 0))
    if total_rows <= 0:
        raise ValueError("Row count for options OI engine must be positive.")

    global_cfg = cfg.get("global", {})
    timezone_name = global_cfg.get("timezone", "UTC")
    resolved_start = _coerce_utc(start_ts, global_cfg.get("default_start_ts"), timezone_name)
    partition_date = resolved_start.tz_convert("UTC").normalize()

    chunk_size = _derive_chunk_size(cfg, total_rows)
    output_path = _resolve_output_path(cfg, partition_date)
    staging_parent = output_path.parent.parent / "_temp_options_oi"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_path = staging_parent / output_path.name

    clock_cfg = {
        "start_ts": resolved_start.isoformat(),
        "inter_event_us": global_cfg.get("inter_event_us", 1000),
    }
    clock = CanonicalClock(clock_cfg)
    seq = SequenceID(global_cfg.get("seed"))
    rng = RNG(seed=global_cfg.get("seed"))

    writer = ParquetWriter(base_path=staging_path, compression=cfg.get("writer", {}).get("compression", "snappy"))

    expiry_candidates = np.array([7, 14, 21, 30, 45, 60, 90])
    expiry_probs = np.array([0.12, 0.16, 0.16, 0.18, 0.14, 0.12, 0.12])

    spot_anchor = float(cfg.get("options", {}).get("spot_anchor", 30000.0))

    produced = 0
    remaining = total_rows

    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        timestamps = [clock.next() for _ in range(batch_size)]
        sequence_ids = seq.next_batch(batch_size)

        expiry_days = rng.choice(expiry_candidates, size=batch_size, p=expiry_probs)
        expiry_days = expiry_days.astype(int)
        expiry_ts = partition_date + pd.to_timedelta(expiry_days, unit="D")

        metrics = _generate_option_metrics(rng, batch_size, spot_anchor, expiry_days)

        df = pd.DataFrame(
            {
                "meta__timestamp": timestamps,
                "meta__sequence_id": sequence_ids,
                "exchange": exchange,
                "symbol": symbol,
                "expiry_ts": expiry_ts,
                "expiry_days": expiry_days.astype(int),
                "strike": metrics["strike"],
                "option_type": metrics["option_type"],
                "open_interest": metrics["open_interest"],
                "oi_change": metrics["oi_change"],
                "volume": metrics["volume"],
                "date": np.full(batch_size, partition_date.strftime("%Y-%m-%d"), dtype=object),
            }
        )

        df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True)
        df["meta__sequence_id"] = df["meta__sequence_id"].astype("int64")
        df["expiry_days"] = df["expiry_days"].astype(int)
        df["open_interest"] = np.maximum(df["open_interest"].astype(int), 0)
        df["oi_change"] = df["oi_change"].astype(int)
        df["volume"] = np.maximum(df["volume"].astype(int), np.abs(df["oi_change"].astype(int)))
        df["date"] = df["date"].astype(str)

        df = df[
            [
                "meta__timestamp",
                "meta__sequence_id",
                "exchange",
                "symbol",
                "expiry_ts",
                "expiry_days",
                "strike",
                "option_type",
                "open_interest",
                "oi_change",
                "volume",
                "date",
            ]
        ]

        writer.write(df)

        produced += batch_size
        remaining -= batch_size

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
        "rows": produced,
        "manifest": manifest,
        "timing": {"seconds": time.time() - t0},
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    res = run_engine()
    print(res)
