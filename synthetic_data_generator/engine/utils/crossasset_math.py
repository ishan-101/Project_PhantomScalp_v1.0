"""
crossasset_math.py

Pure mathematical utilities for cross-asset analysis.
NO config access
NO IO
NO feature naming
NO business logic

Safe for reuse across:
- crossasset_correlation_engine
- crossasset_funding_engine
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# -------------------------------------------------------------------
# Core safety helpers
# -------------------------------------------------------------------

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _safe_std(x: pd.Series) -> float:
    v = x.std(ddof=0)
    return float(v) if v > 0 else np.nan


def _safe_mean(x: pd.Series) -> float:
    return float(x.mean()) if len(x) > 0 else np.nan


def _safe_cov(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.cov(x, y, ddof=0)[0, 1])


# -------------------------------------------------------------------
# Alignment helpers
# -------------------------------------------------------------------

def align_series(
    a: pd.Series,
    b: pd.Series,
    method: str = "inner"
) -> Tuple[pd.Series, pd.Series]:
    """
    Align two time-indexed series safely.

    method:
      - inner : intersection of timestamps
      - outer : union (NaNs preserved)
      - left  : align to a
      - right : align to b
    """
    a = _to_series(a)
    b = _to_series(b)

    if not isinstance(a.index, pd.DatetimeIndex):
        raise ValueError("Series A must have DatetimeIndex")
    if not isinstance(b.index, pd.DatetimeIndex):
        raise ValueError("Series B must have DatetimeIndex")

    joined = pd.concat([a, b], axis=1, join=method)
    joined.columns = ["a", "b"]

    return joined["a"], joined["b"]


def resample_series(
    s: pd.Series,
    rule: str,
    how: str = "last"
) -> pd.Series:
    """
    Resample a series to a new frequency.

    how:
      - last
      - mean
      - sum
    """
    if how == "last":
        return s.resample(rule).last()
    if how == "mean":
        return s.resample(rule).mean()
    if how == "sum":
        return s.resample(rule).sum()

    raise ValueError(f"Unsupported resample method: {how}")


# -------------------------------------------------------------------
# Correlation & covariance
# -------------------------------------------------------------------

def rolling_correlation(
    a: pd.Series,
    b: pd.Series,
    window: int
) -> pd.Series:
    """
    Rolling Pearson correlation.
    """
    a, b = align_series(a, b)
    return a.rolling(window).corr(b)


def ewma_correlation(
    a: pd.Series,
    b: pd.Series,
    alpha: float
) -> pd.Series:
    """
    Exponentially weighted correlation.
    """
    a, b = align_series(a, b)

    mean_a = a.ewm(alpha=alpha).mean()
    mean_b = b.ewm(alpha=alpha).mean()

    cov = (a - mean_a).ewm(alpha=alpha).mean() * (b - mean_b).ewm(alpha=alpha).mean()
    std_a = a.ewm(alpha=alpha).std()
    std_b = b.ewm(alpha=alpha).std()

    return cov / (std_a * std_b)


def covariance(
    a: pd.Series,
    b: pd.Series
) -> float:
    """
    Population covariance.
    """
    a, b = align_series(a, b)
    return _safe_cov(a.dropna(), b.dropna())


def beta(
    a: pd.Series,
    b: pd.Series
) -> float:
    """
    Beta of a relative to b.
    """
    a, b = align_series(a, b)
    var_b = np.var(b, ddof=0)
    if var_b == 0:
        return np.nan
    return _safe_cov(a, b) / var_b


# -------------------------------------------------------------------
# Z-score & normalization
# -------------------------------------------------------------------

def zscore(
    s: pd.Series,
    window: Optional[int] = None
) -> pd.Series:
    """
    Z-score normalization.
    """
    if window:
        mean = s.rolling(window).mean()
        std = s.rolling(window).std(ddof=0)
    else:
        mean = s.mean()
        std = s.std(ddof=0)

    return (s - mean) / std.replace(0, np.nan)


def rolling_zscore(
    s: pd.Series,
    window: int
) -> pd.Series:
    return zscore(s, window=window)


# -------------------------------------------------------------------
# Stability & regime metrics
# -------------------------------------------------------------------

def correlation_stability(
    corr_series: pd.Series
) -> float:
    """
    Measures how stable correlation is over time.
    Lower = more stable.
    """
    corr_series = corr_series.dropna()
    if len(corr_series) < 2:
        return np.nan
    return float(corr_series.std(ddof=0))


def correlation_decay(
    corr_series: pd.Series
) -> float:
    """
    Measures decay between start and end of correlation series.
    """
    corr_series = corr_series.dropna()
    if len(corr_series) < 2:
        return np.nan
    return float(corr_series.iloc[-1] - corr_series.iloc[0])


def rolling_volatility(
    s: pd.Series,
    window: int
) -> pd.Series:
    """
    Rolling volatility (std).
    """
    return s.rolling(window).std(ddof=0)


# -------------------------------------------------------------------
# Funding-specific math (still pure math)
# -------------------------------------------------------------------

def funding_divergence(
    funding_a: pd.Series,
    funding_b: pd.Series
) -> pd.Series:
    """
    Difference between two funding rates.
    """
    a, b = align_series(funding_a, funding_b)
    return a - b


def funding_zscore(
    funding_diff: pd.Series,
    window: int
) -> pd.Series:
    """
    Z-score of funding divergence.
    """
    return rolling_zscore(funding_diff, window)


# -------------------------------------------------------------------
# End of file
# -------------------------------------------------------------------