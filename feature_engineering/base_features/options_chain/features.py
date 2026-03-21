"""Deterministic computation of Options Chain base features per snapshot."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when feature computation fails due to invalid inputs."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _nearest_atm_oi(group: pd.DataFrame, option_type: str) -> float:
    subset = group[group["option_type"] == option_type]
    if subset.empty:
        return 0.0
    spot_value = subset["spot"].iloc[0]
    distance = (subset["strike"] - spot_value).abs()
    # Deterministically select the first strike with minimal distance.
    nearest_idx = distance.idxmin()
    value = subset.loc[nearest_idx, "open_interest"]
    if pd.isna(value):
        return 0.0
    return float(value)


def _best_bid_iv_nearest_maturity(group: pd.DataFrame) -> float:
    if group["time_to_expiry"].isna().all():
        return np.nan
    nearest_expiry = group["time_to_expiry"].min()
    nearest_set = group[group["time_to_expiry"] == nearest_expiry]
    if nearest_set["bid_iv"].isna().all():
        return np.nan
    return float(nearest_set["bid_iv"].max())


def _best_ask_iv_nearest_maturity(group: pd.DataFrame) -> float:
    if group["time_to_expiry"].isna().all():
        return np.nan
    nearest_expiry = group["time_to_expiry"].min()
    nearest_set = group[group["time_to_expiry"] == nearest_expiry]
    if nearest_set["ask_iv"].isna().all():
        return np.nan
    return float(nearest_set["ask_iv"].min())


def _implied_vol_slope(group: pd.DataFrame) -> float:
    # Compute local IV slope using nearest strikes around spot.
    iv_by_strike = group.groupby("strike")["implied_volatility"].mean().dropna()
    if iv_by_strike.empty:
        return 0.0
    spot_value = group["spot"].iloc[0]
    strikes_sorted = iv_by_strike.index.to_series().sort_values()
    lower_strikes = strikes_sorted[strikes_sorted <= spot_value]
    upper_strikes = strikes_sorted[strikes_sorted >= spot_value]

    if lower_strikes.empty or upper_strikes.empty:
        return 0.0

    lower_strike = lower_strikes.iloc[-1]
    upper_strike = upper_strikes.iloc[0]
    if lower_strike == upper_strike:
        return 0.0

    lower_iv = iv_by_strike.loc[lower_strike]
    upper_iv = iv_by_strike.loc[upper_strike]
    slope = (upper_iv - lower_iv) / (upper_strike - lower_strike)
    return float(slope)


def _mean_implied_vol(group: pd.DataFrame) -> float:
    return float(group["implied_volatility"].mean()) if not group["implied_volatility"].isna().all() else np.nan


def _trade_size_by_moneyness(group: pd.DataFrame) -> float:
    spot_series = group["spot"]
    moneyness = safe_divide((group["strike"] - spot_series).abs(), spot_series, allow_zero_division=True, fill_value=0.0)
    weighted = group["volume"].fillna(0.0) * moneyness.fillna(0.0)
    return float(weighted.sum())


def _option_flow_imbalance(group: pd.DataFrame) -> float:
    spot_series = group["spot"].fillna(0.0)
    signed_volume = group["volume"].fillna(0.0) * spot_series
    call_flow = signed_volume.where(group["option_type"] == "call", 0.0).sum()
    put_flow = signed_volume.where(group["option_type"] == "put", 0.0).sum()
    return float(call_flow - put_flow)


def _compute_group_features(group: pd.DataFrame) -> Tuple[float, ...]:
    # Core totals.
    total_calls = float(group.loc[group["option_type"] == "call", "open_interest"].fillna(0.0).sum())
    total_puts = float(group.loc[group["option_type"] == "put", "open_interest"].fillna(0.0).sum())

    nearest_call_oi = _nearest_atm_oi(group, "call")
    nearest_put_oi = _nearest_atm_oi(group, "put")

    call_put_ratio = safe_divide(
        pd.Series(total_calls),
        pd.Series(total_calls + total_puts),
        allow_zero_division=True,
        fill_value=0.5,
    ).iloc[0]
    call_put_ratio = float(call_put_ratio if not np.isnan(call_put_ratio) else 0.5)

    best_bid_iv = _best_bid_iv_nearest_maturity(group)
    best_ask_iv = _best_ask_iv_nearest_maturity(group)

    volume_calls = float(group.loc[group["option_type"] == "call", "volume"].fillna(0.0).sum())
    volume_puts = float(group.loc[group["option_type"] == "put", "volume"].fillna(0.0).sum())

    vol_slope = _implied_vol_slope(group)
    trade_size_moneyness = _trade_size_by_moneyness(group)
    flow_imbalance = _option_flow_imbalance(group)
    mean_iv = _mean_implied_vol(group)

    return (
        nearest_call_oi,
        nearest_put_oi,
        total_calls,
        total_puts,
        call_put_ratio,
        best_bid_iv,
        best_ask_iv,
        volume_calls,
        volume_puts,
        vol_slope,
        trade_size_moneyness,
        flow_imbalance,
        mean_iv,
    )


def compute_options_chain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base Options Chain features using current and previous snapshots only.

    Args:
        df: Input option chain snapshot data with per-option rows. Expected columns:
            ["ts", "symbol", "spot", "option_type", "strike", "open_interest", "volume",
             "implied_volatility", "bid_iv", "ask_iv", "time_to_expiry"].

    Returns:
        DataFrame aggregated to one row per (symbol, ts) containing the 15 Options Chain features.

    Raises:
        FeatureComputationError: If required inputs are missing or option_type values are invalid.
    """

    required_columns = [
        "ts",
        "symbol",
        "spot",
        "option_type",
        "strike",
        "open_interest",
        "volume",
        "implied_volatility",
        "bid_iv",
        "ask_iv",
        "time_to_expiry",
    ]
    _require_columns(df, required_columns)

    invalid_types = set(df["option_type"].dropna().unique()) - {"call", "put"}
    if invalid_types:
        raise FeatureComputationError(f"Unexpected option_type values: {sorted(invalid_types)}")

    working = df.copy()
    working.sort_values(["symbol", "ts", "time_to_expiry", "strike"], inplace=True)

    # Spot price forward-filled within each symbol, then filled with zero.
    working["spot"] = (
        working.groupby("symbol")["spot"].ffill().fillna(0.0).astype("float32")
    )

    grouped = working.groupby(["symbol", "ts"], sort=True)

    feature_records = grouped.apply(_compute_group_features)
    feature_df = feature_records.apply(pd.Series)
    feature_df.columns = [
        "opt__nearest_oi_call",
        "opt__nearest_oi_put",
        "opt__total_oi_calls",
        "opt__total_oi_puts",
        "opt__call_put_oi_ratio",
        "opt__best_bid_iv",
        "opt__best_ask_iv",
        "opt__volume_calls",
        "opt__volume_puts",
        "opt__implied_vol_slope",
        "opt__trade_size_by_moneyness_proxy",
        "opt__option_flow_imbalance",
        "_mean_implied_vol",
    ]

    feature_df["opt__spot"] = grouped["spot"].first().astype("float32")

    total_oi = feature_df["opt__total_oi_calls"] + feature_df["opt__total_oi_puts"]
    total_oi_prev = total_oi.groupby(level=0).shift(1)
    feature_df["opt__oi_change"] = (total_oi - total_oi_prev).fillna(0.0)

    # Best bid/ask IV forward-filled from previous snapshot if nearest maturity data is missing.
    feature_df["opt__best_bid_iv"] = (
        feature_df["opt__best_bid_iv"].groupby(level=0).ffill().fillna(0.0)
    )
    feature_df["opt__best_ask_iv"] = (
        feature_df["opt__best_ask_iv"].groupby(level=0).ffill().fillna(0.0)
    )

    # IV crush detector: true when mean IV drops more than 10% relative to previous snapshot.
    mean_iv_prev = feature_df["_mean_implied_vol"].groupby(level=0).shift(1)
    iv_crush = (feature_df["_mean_implied_vol"] < mean_iv_prev * 0.9) & mean_iv_prev.notna()
    feature_df["opt__iv_crush_detector"] = iv_crush.fillna(False)

    feature_df.drop(columns=["_mean_implied_vol"], inplace=True)

    # Call/put OI ratio fallbacks.
    feature_df["opt__call_put_oi_ratio"] = (
        feature_df["opt__call_put_oi_ratio"].fillna(0.5).clip(0.0, 1.0)
    )

    # Null policies for zero-if-missing features.
    zero_fill_cols = [
        "opt__nearest_oi_call",
        "opt__nearest_oi_put",
        "opt__total_oi_calls",
        "opt__total_oi_puts",
        "opt__volume_calls",
        "opt__volume_puts",
        "opt__option_flow_imbalance",
        "opt__trade_size_by_moneyness_proxy",
    ]
    for col in zero_fill_cols:
        feature_df[col] = feature_df[col].fillna(0.0)

    # Computed else zero fallbacks.
    computed_zero_cols = ["opt__oi_change", "opt__implied_vol_slope"]
    for col in computed_zero_cols:
        feature_df[col] = feature_df[col].fillna(0.0)

    feature_df["opt__spot"] = feature_df["opt__spot"].fillna(0.0)

    # Explicit dtypes.
    float_cols = [
        "opt__spot",
        "opt__nearest_oi_call",
        "opt__nearest_oi_put",
        "opt__total_oi_calls",
        "opt__total_oi_puts",
        "opt__call_put_oi_ratio",
        "opt__best_bid_iv",
        "opt__best_ask_iv",
        "opt__oi_change",
        "opt__volume_calls",
        "opt__volume_puts",
        "opt__trade_size_by_moneyness_proxy",
        "opt__implied_vol_slope",
        "opt__option_flow_imbalance",
    ]
    for col in float_cols:
        feature_df[col] = feature_df[col].astype("float32")

    feature_df["opt__iv_crush_detector"] = feature_df["opt__iv_crush_detector"].astype(bool)

    # Reset index to expose symbol and ts alongside computed features.
    output = feature_df.reset_index()

    expected_columns = [
        "symbol",
        "ts",
        "opt__spot",
        "opt__nearest_oi_call",
        "opt__nearest_oi_put",
        "opt__total_oi_calls",
        "opt__total_oi_puts",
        "opt__call_put_oi_ratio",
        "opt__best_bid_iv",
        "opt__best_ask_iv",
        "opt__oi_change",
        "opt__volume_calls",
        "opt__volume_puts",
        "opt__iv_crush_detector",
        "opt__trade_size_by_moneyness_proxy",
        "opt__implied_vol_slope",
        "opt__option_flow_imbalance",
    ]
    missing = set(expected_columns) - set(output.columns)
    if missing:
        raise FeatureComputationError(f"Missing computed features: {sorted(missing)}")

    return output[expected_columns]


__all__ = ["compute_options_chain_features", "FeatureComputationError"]
