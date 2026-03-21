"""Cross-asset funding engine.

Generates synthetic cross-asset funding relationships using math utilities
from ``engine/utils/crossasset_math.py`` and writes a single partitioned
parquet output derived entirely from the central config.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from synthetic_data_generator.engine.config.loader import ConfigError, load_config
from synthetic_data_generator.engine.utils import crossasset_math as cam
from synthetic_data_generator.engine.utils.io_writer import ParquetWriter
from synthetic_data_generator.engine.utils.sequence_id import SequenceID


class FundingEngineError(Exception):
    """Domain-specific errors for the cross-asset funding engine."""


def _tz_aware_timestamp(value: str, timezone: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if timezone.upper() != "UTC":
        ts = ts.tz_convert(timezone)
    return ts


def _derive_paths(cfg: Dict, partition_date: str) -> Path:
    try:
        base = cfg["paths"]["base"]
        rel = cfg["paths"]["crossasset_funding"]
    except KeyError as exc:
        raise FundingEngineError(f"Missing path configuration: {exc}") from exc
    return Path(base) / rel / f"date={partition_date}"


def _generate_mean_reverting_series(length: int, rng: np.random.Generator) -> np.ndarray:
    series = np.zeros(length, dtype=np.float64)
    mean_reversion = 0.98
    shock_scale = 0.0005
    for i in range(1, length):
        noise = rng.normal(loc=0.0, scale=shock_scale)
        series[i] = mean_reversion * series[i - 1] + noise
    return np.clip(series, -0.01, 0.01)


def _assign_regime(zscores: pd.Series) -> pd.Series:
    thresholds = {
        "LOW": zscores <= -1.0,
        "HIGH": zscores >= 1.0,
    }
    regime = pd.Series(np.where(thresholds["LOW"], "LOW", "NEUTRAL"), index=zscores.index)
    regime = pd.Series(np.where(thresholds["HIGH"], "HIGH", regime), index=zscores.index)
    return regime


def generate_crossasset_funding(rows_override: Optional[int] = None, start_ts: Optional[str] = None) -> Dict:
    cfg = load_config()
    t0 = time.perf_counter()

    try:
        total_rows = int(rows_override or cfg["rows"]["crossasset_funding"])
        timezone = cfg["global"]["timezone"]
        seed = int(cfg["global"]["seed"])
        default_start_ts = cfg["global"]["default_start_ts"]
        compression = cfg["writer"].get("compression", "snappy")
    except KeyError as exc:
        raise FundingEngineError(f"Missing required configuration key: {exc}") from exc

    if total_rows <= 0:
        raise FundingEngineError("Row count must be positive")

    start_timestamp = _tz_aware_timestamp(start_ts or default_start_ts, timezone)
        # Keep all generated rows within a single calendar day so that the partition
    # name (date=YYYY-MM-DD) matches the data inside the file. Minute-level
    # spacing would push 5,000 rows across multiple days, so use second-level
    # spacing instead.
    timestamps = pd.date_range(start=start_timestamp, periods=total_rows, freq="s", tz=start_timestamp.tz)

    rng = np.random.default_rng(seed)
    seq = SequenceID(seed=seed, start=1)

    base_rates = _generate_mean_reverting_series(total_rows, rng)
    quote_rates = _generate_mean_reverting_series(total_rows, rng)

    base_series = pd.Series(base_rates, index=timestamps)
    quote_series = pd.Series(quote_rates, index=timestamps)

    funding_diff = cam.funding_divergence(base_series, quote_series).astype(np.float64)
    funding_zscore = cam.funding_zscore(funding_diff, window=50).fillna(0.0).astype(np.float64)
    funding_volatility = cam.rolling_volatility(funding_diff, window=50).fillna(0.0).astype(np.float64)

    df = pd.DataFrame(
        {
            "meta__timestamp": timestamps,
            "meta__sequence_id": pd.Series(seq.next_batch(total_rows), index=timestamps, dtype=np.int64).values,
            "date": timestamps.tz_convert("UTC").strftime("%Y-%m-%d"),
            "base_symbol": "BTC",
            "quote_symbol": "ETH",
            "funding_rate_base": base_series.values.astype(np.float64),
            "funding_rate_quote": quote_series.values.astype(np.float64),
            "funding_diff": funding_diff.values,
            "funding_diff_zscore": funding_zscore.values,
            "funding_volatility": funding_volatility.values,
            "funding_regime": _assign_regime(funding_zscore).values,
        }
    )

    if df.isna().any().any():
        raise FundingEngineError("NaNs detected in generated data")

    partition_date = df.loc[0, "date"]
    output_dir = _derive_paths(cfg, partition_date)
    output_path = output_dir / "crossasset_funding.parquet"

    writer = ParquetWriter(output_path, compression=compression)
    writer.write(df, append=False)

    elapsed = time.perf_counter() - t0
    manifest = writer.get_manifest()

    return {
        "rows": int(total_rows),
        "manifest": manifest,
        "timing": {"seconds": float(elapsed)},
    }


__all__ = ["generate_crossasset_funding", "FundingEngineError"]
