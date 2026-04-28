"""Rolling window primitives with explicit validation and alignment."""

from __future__ import annotations

import pandas as pd


def _validate_window(window: int) -> None:
    if window <= 0:
        raise ValueError("window must be a positive integer")


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean with full-window requirement."""
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling standard deviation with full-window requirement."""
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).std()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling minimum with full-window requirement."""
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).min()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling maximum with full-window requirement."""
    _validate_window(window)
    return series.rolling(window=window, min_periods=window).max()
