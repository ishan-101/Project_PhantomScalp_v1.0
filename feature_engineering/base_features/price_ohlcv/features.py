"""Deterministic computation of Price / OHLCV base features per record."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


# Named thresholds must be provided by callers to avoid hidden magic numbers.
class FeatureComputationError(RuntimeError):
    """Raised when feature computation cannot proceed due to invalid inputs."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def compute_price_ohlcv_features(
    df: pd.DataFrame,
    *,
    jump_threshold: float,
    sweep_volume_threshold: float,
    sweep_spread_threshold: float,
) -> pd.DataFrame:
    """Compute base Price / OHLCV features using only current and previous record data.

    Args:
        df: Input DataFrame already sorted, schema-aligned, and dtype-enforced.
        jump_threshold: Absolute tick return threshold for the jump flag (must be positive).
        sweep_volume_threshold: Volume threshold used for liquidity sweep detection.
        sweep_spread_threshold: Absolute spread change threshold for liquidity sweep detection.

    Returns:
        A new DataFrame containing the original columns plus the 25 base features.

    Raises:
        FeatureComputationError: If required inputs are missing or thresholds are invalid.
    """

    if jump_threshold <= 0:
        raise FeatureComputationError("jump_threshold must be positive.")
    if sweep_volume_threshold <= 0:
        raise FeatureComputationError("sweep_volume_threshold must be positive.")
    if sweep_spread_threshold <= 0:
        raise FeatureComputationError("sweep_spread_threshold must be positive.")

    required_columns = [
        "price__last",
        "price__bid",
        "price__ask",
        "bid_size",
        "ask_size",
        "ohlcv__open",
        "ohlcv__high",
        "ohlcv__low",
        "ohlcv__close",
        "volume__tick",
        "trade_count",
        "buy_volume",
        "sell_volume",
        "vwap__in_record",
        "prev_price__last",
        "prev_price__bid",
        "prev_price__ask",
        "prev_price__mid",
    ]
    _require_columns(df, required_columns)

    output = df.copy()

    # Core prices with explicit null handling.
    output["price__last"] = output["price__last"].ffill().fillna(0.0)
    output["price__bid"] = output["price__bid"].ffill().fillna(0.0)
    output["price__ask"] = output["price__ask"].ffill().fillna(0.0)

    # Mid price from bid/ask, otherwise forward-fill then default.
    mid_computed = (output["price__bid"] + output["price__ask"]) / 2
    output["price__mid"] = mid_computed.where(~mid_computed.isna())
    output["price__mid"] = output["price__mid"].ffill().fillna(0.0)

    # OHLC values forward-filled when trade prices are missing.
    for col in ("ohlcv__open", "ohlcv__high", "ohlcv__low", "ohlcv__close"):
        output[col] = output[col].ffill().fillna(0.0)

    # Volume and trade count with explicit zero fill.
    output["volume__tick"] = output["volume__tick"].fillna(0.0)
    output["trade_count"] = output["trade_count"].fillna(0).astype("int32")

    # Buy/sell imbalance as the signed difference between buy and sell volume.
    imbalance_raw = output["buy_volume"].fillna(0.0) - output["sell_volume"].fillna(0.0)
    output["volume__buy_sell_imbalance"] = imbalance_raw

    # VWAP uses provided value; fallback to last price when unavailable.
    output["vwap__in_record"] = output["vwap__in_record"].fillna(output["price__last"]).fillna(0.0)

    # Spread computed from bid/ask with zero fallback.
    spread = (output["price__ask"] - output["price__bid"]).fillna(0.0)
    output["spread__l1"] = spread

    # Returns and direction using previous record fields (assumed aligned).
    prev_last = output["prev_price__last"].fillna(0.0)
    prev_bid = output["prev_price__bid"].fillna(0.0)
    prev_ask = output["prev_price__ask"].fillna(0.0)
    prev_mid = output["prev_price__mid"].fillna(0.0)

    tick_return = safe_divide(
        output["price__last"] - prev_last,
        prev_last,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)
    output["tick_return"] = tick_return

    price_move = output["price__last"] - prev_last
    output["price__tick_direction"] = np.sign(price_move).astype("int32")

    output["return__bid_change"] = safe_divide(
        output["price__bid"] - prev_bid,
        prev_bid,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["return__ask_change"] = safe_divide(
        output["price__ask"] - prev_ask,
        prev_ask,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    current_mid = output["price__mid"]
    output["return__mid_change"] = safe_divide(
        current_mid - prev_mid,
        prev_mid,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    # Micro-volatility proxies.
    output["price__micro_volatility"] = tick_return.abs()
    output["price__near_term_return_volatility"] = output["return__mid_change"].abs()

    # Jump flag on absolute tick return exceeding configured threshold.
    output["price__jump_flag"] = (output["price__micro_volatility"] > jump_threshold).fillna(False)

    # Slippage estimate derived from spread relative to mid price.
    output["exec__slippage_estimate"] = safe_divide(
        spread,
        current_mid.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    # Depth-weighted fair value using opposing side sizes; fallback to mid price.
    depth_weighted = safe_divide(
        output["price__bid"] * output["ask_size"] + output["price__ask"] * output["bid_size"],
        output["bid_size"] + output["ask_size"],
        allow_zero_division=True,
        fill_value=np.nan,
    )
    output["price__micro_fair"] = depth_weighted.fillna(current_mid).fillna(0.0)

    # Imbalance-adjusted return.
    output["price__imbalance_adjusted_return"] = tick_return * output["volume__buy_sell_imbalance"]

    # Liquidity sweep flag based on size and spread change magnitudes.
    prev_spread = (prev_ask - prev_bid).fillna(0.0)
    spread_change = (spread - prev_spread).abs()
    output["price__tick_sweep_flag"] = (
        (output["volume__tick"] >= sweep_volume_threshold)
        & (spread_change >= sweep_spread_threshold)
    ).fillna(False)

    return output
