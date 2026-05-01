# feature_engineering/utils/futures_helpers.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .math_helpers import safe_divide
from .rolling import rolling_mean, rolling_std


# ---------------------------------------------------------
# BASIS
# ---------------------------------------------------------
def compute_basis(perp_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    return safe_divide(perp_price - spot_price, spot_price, allow_zero_division=True, fill_value=0.0)


def compute_basis_pct(perp_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    return compute_basis(perp_price, spot_price) * 100.0


# ---------------------------------------------------------
# Z-SCORE (SAFE)
# ---------------------------------------------------------
def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = rolling_mean(series, window)
    std = rolling_std(series, window)
    std = std.replace(0, np.nan)
    z = (series - mean) / std
    return z


# ---------------------------------------------------------
# FLAGS
# ---------------------------------------------------------
def extreme_flag(series: pd.Series, threshold: float) -> pd.Series:
    """
    Returns int32 0/1 flag based on absolute threshold.
    """
    return (series.abs() > threshold).astype("int32")


def regime_flag(z: pd.Series, pos_th: float = 1.0, neg_th: float = -1.0) -> pd.Series:
    """
    Ternary regime: -1, 0, 1 (int32)
    """
    return (z > pos_th).astype("int32") - (z < neg_th).astype("int32")


# ---------------------------------------------------------
# SAFE PRODUCT (used in interactions)
# ---------------------------------------------------------
def safe_product(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a * b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out