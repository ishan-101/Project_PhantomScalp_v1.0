"""Feature computation for futures open-interest submodule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.utils.feature_helpers import (
    apply_schema_null_policy,
    clean_compute_output,
    finalize_frame,
)

SCHEMA_PATH = Path(__file__).with_name("schema.json")

with SCHEMA_PATH.open("r", encoding="utf-8") as f:
    _SCHEMA = json.load(f)

FEATURE_COLUMNS = list(_SCHEMA.keys())
FEATURE_DTYPES = {col: meta["dtype"] for col, meta in _SCHEMA.items()}


def _apply_policy(df: pd.DataFrame, column: str) -> pd.DataFrame:
    policy = _SCHEMA[column]["null_policy"]
    out, _ = apply_schema_null_policy(df, [column], policy)
    return out


def compute_features(
    snapshot: pd.DataFrame,
    upstream_features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Compute open-interest features defined by schema.json."""
    del upstream_features
    window = int(config.get("rolling_window", 50))
    if window <= 0:
        raise ValueError("rolling_window must be positive")

    oi = snapshot["open_interest"].astype("float64")
    price = snapshot["price"].astype("float64")
    volume = snapshot["volume"].astype("float64")

    oi_change = oi.diff()
    oi_velocity = oi_change.diff()
    oi_acceleration = oi_velocity.diff()

    rolling_mean = oi.rolling(window=window, min_periods=2).mean()
    rolling_std = oi.rolling(window=window, min_periods=2).std(ddof=0)
    oi_zscore = (oi - rolling_mean) / rolling_std.replace(0.0, np.nan)

    oi_turnover = volume / oi.replace(0.0, np.nan)

    price_return = price.diff()
    oi_price_divergence = np.sign(oi_change) - np.sign(price_return)

    oi_change_pos = oi_change.clip(lower=0.0)
    oi_change_neg = (-oi_change.clip(upper=0.0))
    oi_open_close_ratio = oi_change_pos / oi_change_neg.replace(0.0, np.nan)

    oi_vol = oi_change.rolling(window=window, min_periods=2).std(ddof=0)
    px_vol = price_return.rolling(window=window, min_periods=2).std(ddof=0)
    denom = oi_vol + px_vol
    oi_price_divergence_strength = oi_price_divergence.abs() / denom.replace(0.0, np.nan)

    out = pd.DataFrame(index=snapshot.index)
    out[FEATURE_COLUMNS[0]] = oi
    out[FEATURE_COLUMNS[1]] = oi_change
    out[FEATURE_COLUMNS[2]] = oi_velocity
    out[FEATURE_COLUMNS[3]] = oi_acceleration
    out[FEATURE_COLUMNS[4]] = oi_zscore
    out[FEATURE_COLUMNS[5]] = oi_price_divergence
    out[FEATURE_COLUMNS[6]] = oi_price_divergence_strength
    out[FEATURE_COLUMNS[7]] = oi_turnover
    out[FEATURE_COLUMNS[8]] = oi_open_close_ratio

    out = clean_compute_output(out, FEATURE_COLUMNS)

    for col in FEATURE_COLUMNS:
        out = _apply_policy(out, col)

    out = out[FEATURE_COLUMNS]
    out = finalize_frame(out, FEATURE_DTYPES)

    if out.isna().any().any():
        raise ValueError("Feature output contains nulls after processing")

    return out
