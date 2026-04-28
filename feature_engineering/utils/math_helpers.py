"""Mathematical helper functions for feature engineering primitives."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    fill_value: Number = np.nan,
    allow_zero_division: bool = False,
) -> pd.Series:
    """Safely divide two series with explicit zero-handling policy."""
    if numerator.index.difference(denominator.index).any() or denominator.index.difference(
        numerator.index
    ).any():
        raise ValueError("Numerator and denominator indices must align exactly.")

    zero_denominator = denominator == 0
    if zero_denominator.any() and not allow_zero_division:
        raise ZeroDivisionError("Denominator contains zeros; set allow_zero_division to True.")

    result = numerator / denominator
    if allow_zero_division:
        result = result.mask(zero_denominator, fill_value)
    return result


def log_returns(series: pd.Series, *, periods: int = 1) -> pd.Series:
    """Compute logarithmic returns with validation against non-positive inputs."""
    if periods <= 0:
        raise ValueError("periods must be positive for log returns.")
    if (series <= 0).any():
        raise ValueError("Log returns require strictly positive series values.")
    log_series = np.log(series)
    return log_series.diff(periods=periods)


def simple_difference(series: pd.Series, *, periods: int = 1) -> pd.Series:
    """Compute simple difference over the specified lag."""
    if periods <= 0:
        raise ValueError("periods must be positive for simple differences.")
    return series.diff(periods=periods)


def slope(series: pd.Series, *, periods: int = 1) -> pd.Series:
    """Compute per-period slope as difference divided by elapsed periods."""
    if periods <= 0:
        raise ValueError("periods must be positive for slope computation.")
    return simple_difference(series, periods=periods) / periods
