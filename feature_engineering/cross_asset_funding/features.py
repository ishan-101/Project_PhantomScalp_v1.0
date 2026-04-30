"""Deterministic computation of Cross-Asset / Funding base features."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from feature_engineering.utils.rolling import rolling_mean


class FeatureComputationError(RuntimeError):
    """Raised when feature computation fails due to invalid inputs."""


REQUIRED_COLUMNS: List[str] = ["ts", "btc_spot", "btc_perpetual", "eth_spot", "funding_rate", "dxy_index"]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _validate_time_deltas(timestamps: pd.Series) -> pd.Series:
    if timestamps.isna().any():
        raise FeatureComputationError("Timestamp column contains null values.")
    deltas = timestamps.diff().dt.total_seconds()
    if (deltas <= 0).iloc[1:].any():
        raise FeatureComputationError("Timestamps must be strictly increasing to compute velocities.")
    return deltas


def compute_cross_asset_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cross-Asset / Funding base features using snapshot-causal logic.

    Args:
        df: Input DataFrame with columns:
            ["ts", "btc_spot", "btc_perpetual", "eth_spot", "funding_rate", "dxy_index"].
            Rows must be ordered or orderable by timestamp.

    Returns:
        DataFrame with the nine cross-asset / funding features alongside ts.

    Raises:
        FeatureComputationError: If required inputs are missing or validation fails.
    """
    _require_columns(df, REQUIRED_COLUMNS)

    working = df.copy()
    working = working.sort_values("ts").reset_index(drop=True)

    time_delta_seconds = _validate_time_deltas(working["ts"])

    # Fundamental inputs cast to float for deterministic math.
    for col in ["btc_spot", "btc_perpetual", "eth_spot", "funding_rate", "dxy_index"]:
        working[col] = working[col].astype("float64")

    # Core spreads and bases.
    btc_eth_spread = working["btc_spot"] - working["eth_spot"]
    perp_spot_basis = working["btc_perpetual"] - working["btc_spot"]

    # Funding rate forward-filled only (no smoothing).
    funding_rate_ffill = working["funding_rate"].ffill()

    # Return series for correlation proxy (log returns to stay scale-free).
    btc_return = np.log(working["btc_spot"]).diff()
    dxy_return = np.log(working["dxy_index"]).diff()

    corr_window = 24
    btc_dxy_corr_proxy = (
        btc_return.rolling(window=corr_window, min_periods=corr_window)
        .corr(dxy_return)
    )

    # Funding rolling mean over 8 backward-looking snapshots.
    funding_8h_avg = rolling_mean(funding_rate_ffill, window=8)

    basis_change = perp_spot_basis.diff()

    # Risk on/off flag: positive funding AND positive basis AND positive correlation proxy.
    risk_on_off_flag = (funding_rate_ffill > 0) & (perp_spot_basis > 0) & (btc_dxy_corr_proxy > 0)

    # Normalize basis change per hour to avoid magnitude inflation when cadence changes.
    basis_velocity = basis_change / time_delta_seconds
    basis_velocity_per_hour = basis_velocity * 3600.0

    correlation_instability = btc_dxy_corr_proxy.diff().abs()

    features = pd.DataFrame(
        {
            "ts": working["ts"],
            "cross__btc_eth_spread": btc_eth_spread,
            "cross__perp_spot_basis": perp_spot_basis,
            "cross__funding_rate": funding_rate_ffill,
            "cross__btc_dxy_corr_proxy": btc_dxy_corr_proxy,
            "cross__funding_8h_rolling_avg": funding_8h_avg,
            "cross__basis_change": basis_change,
            "cross__risk_on_off_flag": risk_on_off_flag,
            "cross__perp_basis_velocity": basis_velocity_per_hour,
            "cross__correlation_instability": correlation_instability,
        }
    )

    # Drop rows lacking full backward-looking windows to avoid silent filling.
    complete_rows = features.dropna().copy()
    if complete_rows.empty:
        raise FeatureComputationError("Insufficient history to compute full-window features without nulls.")

    float_cols = [
        "cross__btc_eth_spread",
        "cross__perp_spot_basis",
        "cross__funding_rate",
        "cross__btc_dxy_corr_proxy",
        "cross__funding_8h_rolling_avg",
        "cross__basis_change",
        "cross__perp_basis_velocity",
        "cross__correlation_instability",
    ]
    for col in float_cols:
        complete_rows[col] = complete_rows[col].astype("float32")
        if not np.isfinite(complete_rows[col]).all():
            raise FeatureComputationError(f"Non-finite values detected in {col}.")

    complete_rows["cross__risk_on_off_flag"] = complete_rows["cross__risk_on_off_flag"].astype(bool)

    if complete_rows.isna().any().any():
        raise FeatureComputationError("Nulls remain after computation; aborting to avoid silent fills.")

    expected_columns = ["ts"] + [col for col in complete_rows.columns if col.startswith("cross__")]
    return complete_rows[expected_columns]


__all__ = ["compute_cross_asset_funding_features", "FeatureComputationError"]
