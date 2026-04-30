"""Compute base microstructure L2/L3 features per snapshot using current and prior state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when feature computation cannot proceed due to invalid inputs."""


@dataclass
class OrderBookColumns:
    """Resolved column collections for bid/ask levels."""

    bid_prices: list[str]
    ask_prices: list[str]
    bid_sizes: list[str]
    ask_sizes: list[str]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _resolve_level_columns(df: pd.DataFrame) -> OrderBookColumns:
    """Resolve bid/ask level columns based on naming convention bid/ask_(price|size)_N."""

    bid_price_cols = sorted(
        [c for c in df.columns if c.startswith("bid_price_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    ask_price_cols = sorted(
        [c for c in df.columns if c.startswith("ask_price_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    bid_size_cols = sorted(
        [c for c in df.columns if c.startswith("bid_size_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    ask_size_cols = sorted(
        [c for c in df.columns if c.startswith("ask_size_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    if not bid_price_cols or not ask_price_cols or not bid_size_cols or not ask_size_cols:
        raise FeatureComputationError("Order book level columns are required to compute microstructure features.")
    if not (len(bid_price_cols) == len(ask_price_cols) == len(bid_size_cols) == len(ask_size_cols)):
        raise FeatureComputationError("Bid/ask price and size level counts must match exactly.")

    return OrderBookColumns(
        bid_prices=bid_price_cols,
        ask_prices=ask_price_cols,
        bid_sizes=bid_size_cols,
        ask_sizes=ask_size_cols,
    )


def _linear_slope(sizes: np.ndarray) -> float:
    """Compute slope of size with respect to level index using analytical OLS solution."""

    n = sizes.size
    if n <= 1:
        return 0.0
    indices = np.arange(n, dtype=float)
    mean_i = indices.mean()
    mean_s = sizes.mean()
    denom = ((indices - mean_i) ** 2).sum()
    if denom == 0:
        return 0.0
    numer = ((indices - mean_i) * (sizes - mean_s)).sum()
    return float(numer / denom)


def compute_microstructure_l2_l3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 28 microstructure L2/L3 base features using current and previous snapshots."""

    required_base = ["top_bid", "top_ask", "spread"]
    _require_columns(df, required_base)
    columns = _resolve_level_columns(df)

    output = df.copy()

    bid_sizes = output[columns.bid_sizes]
    ask_sizes = output[columns.ask_sizes]
    bid_prices = output[columns.bid_prices]
    ask_prices = output[columns.ask_prices]

    # Top-level values.
    top_bid_size = bid_sizes.iloc[:, 0].fillna(0.0)
    top_ask_size = ask_sizes.iloc[:, 0].fillna(0.0)
    spread = (output["top_ask"] - output["top_bid"]).fillna(0.0)
    mid_price = ((output["top_bid"].fillna(0.0) + output["top_ask"].fillna(0.0)) / 2).fillna(0.0)

    # Aggregate depth.
    total_depth_bid = bid_sizes.fillna(0.0).sum(axis=1)
    total_depth_ask = ask_sizes.fillna(0.0).sum(axis=1)
    total_depth = total_depth_bid + total_depth_ask

    imbalance = safe_divide(
        total_depth_bid - total_depth_ask,
        total_depth_bid + total_depth_ask,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    # Distance weights favoring near-touch liquidity (1 / (level + 1)).
    level_weights = np.reciprocal(np.arange(1, bid_sizes.shape[1] + 1, dtype=float))
    weighted_bid = (bid_sizes.fillna(0.0) * level_weights).sum(axis=1)
    weighted_ask = (ask_sizes.fillna(0.0) * level_weights).sum(axis=1)
    distance_weighted_total = weighted_bid + weighted_ask
    depth_imbalance_by_distance = safe_divide(
        weighted_bid - weighted_ask,
        distance_weighted_total,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    signed_pressure = safe_divide(
        imbalance * total_depth,
        spread.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    imbalance_spread_adj = (
        imbalance
        * safe_divide(
            (mid_price.replace(0, np.nan)),
            (spread + mid_price).replace(0, np.nan),
            allow_zero_division=True,
            fill_value=0.0,
        )
    ).fillna(0.0)

    # Level decay as ratio of top liquidity to average depth beyond the first level.
    if bid_sizes.shape[1] > 1:
        bid_tail_mean = bid_sizes.iloc[:, 1:].fillna(0.0).mean(axis=1)
        ask_tail_mean = ask_sizes.iloc[:, 1:].fillna(0.0).mean(axis=1)
    else:
        bid_tail_mean = pd.Series(0.0, index=output.index)
        ask_tail_mean = pd.Series(0.0, index=output.index)
    level_decay_rate = (
        safe_divide(top_bid_size, bid_tail_mean.replace(0, np.nan), allow_zero_division=True, fill_value=0.0)
        + safe_divide(top_ask_size, ask_tail_mean.replace(0, np.nan), allow_zero_division=True, fill_value=0.0)
    ) / 2

    orderbook_gap = spread

    bid_slopes = bid_sizes.fillna(0.0).apply(lambda row: _linear_slope(row.to_numpy()), axis=1)
    ask_slopes = ask_sizes.fillna(0.0).apply(lambda row: _linear_slope(row.to_numpy()), axis=1)
    curvature = bid_slopes - ask_slopes

    depth_skew = safe_divide(
        bid_sizes.iloc[:, : min(3, bid_sizes.shape[1])].fillna(0.0).sum(axis=1)
        - ask_sizes.iloc[:, : min(3, ask_sizes.shape[1])].fillna(0.0).sum(axis=1),
        total_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    size_vector = pd.concat([bid_sizes.fillna(0.0), ask_sizes.fillna(0.0)], axis=1)
    size_sum = size_vector.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = size_vector.div(size_sum, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        entropy_components = probs.mul(np.log(probs.replace(0, np.nan)), axis=0).fillna(0.0)
    book_entropy = (-entropy_components.sum(axis=1)).fillna(0.0)

    cancellation_rate = safe_divide(
        output.get("cancellations", pd.Series(0.0, index=output.index)).fillna(0.0),
        total_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    new_order_rate = safe_divide(
        output.get("new_orders", pd.Series(0.0, index=output.index)).fillna(0.0),
        total_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    prev_top_bid_size = top_bid_size.shift(1).fillna(0.0)
    prev_top_ask_size = top_ask_size.shift(1).fillna(0.0)
    prev_spread = spread.shift(1).fillna(0.0)
    prev_total_depth = total_depth.shift(1).fillna(0.0)
    prev_imbalance = imbalance.shift(1).fillna(0.0)
    prev_signed_pressure = signed_pressure.shift(1).fillna(0.0)

    top_turnover = safe_divide(
        (top_bid_size - prev_top_bid_size).abs() + (top_ask_size - prev_top_ask_size).abs(),
        (prev_top_bid_size + prev_top_ask_size),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    queue_position_change_bid = (top_bid_size - prev_top_bid_size).fillna(0.0)
    queue_position_change_ask = (top_ask_size - prev_top_ask_size).fillna(0.0)

    queue_resilience = safe_divide(
        (top_bid_size + top_ask_size) - (prev_top_bid_size + prev_top_ask_size),
        prev_spread.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)
    queue_resilience = queue_resilience.clip(lower=0.0)

    queue_imbalance_momentum = (imbalance - prev_imbalance).fillna(0.0)

    hidden_size = output.get("hidden_volume", pd.Series(0.0, index=output.index)).fillna(0.0)
    visible_touch = (top_bid_size + top_ask_size).replace(0, np.nan)
    hidden_to_visible_ratio = safe_divide(
        hidden_size,
        visible_touch,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)
    hidden_liquidity_indicator = (hidden_to_visible_ratio > 1.0).fillna(False)

    # Liquidity void based on large price gaps between adjacent levels relative to top spread.
    price_gaps_bid = bid_prices.diff(axis=1).abs().fillna(0.0)
    price_gaps_ask = ask_prices.diff(axis=1).abs().fillna(0.0)
    large_gap = pd.concat([price_gaps_bid, price_gaps_ask], axis=1).max(axis=1)
    liquidity_void_flag = (large_gap > (spread * 2)).fillna(False)

    bid_dominance_factor = (
        safe_divide(
            bid_slopes - ask_slopes,
            (bid_slopes.abs() + ask_slopes.abs()),
            allow_zero_division=True,
            fill_value=0.0,
        )
        .clip(-1, 1)
        .fillna(0.0)
    )

    microstructure_pressure_velocity = (signed_pressure - prev_signed_pressure).fillna(0.0)

    depth_elasticity = safe_divide(
        spread - prev_spread,
        total_depth - prev_total_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["ob__total_depth_bid"] = total_depth_bid
    output["ob__total_depth_ask"] = total_depth_ask
    output["ob__top_level_size_bid"] = top_bid_size
    output["ob__top_level_size_ask"] = top_ask_size
    output["ob__imbalance"] = imbalance
    output["ob__signed_book_pressure"] = signed_pressure
    output["ob__imbalance_spread_adj"] = imbalance_spread_adj
    output["ob__depth_imbalance_by_distance"] = depth_imbalance_by_distance
    output["ob__level_decay_rate"] = level_decay_rate
    output["ob__orderbook_gap"] = orderbook_gap
    output["ob__bid_slope"] = bid_slopes
    output["ob__ask_slope"] = ask_slopes
    output["ob__curvature"] = curvature
    output["ob__depth_skew"] = depth_skew
    output["ob__book_entropy"] = book_entropy
    output["ob__cancellation_rate"] = cancellation_rate
    output["ob__new_order_rate"] = new_order_rate
    output["ob__top_of_book_turnover"] = top_turnover
    output["ob__queue_position_change_bid"] = queue_position_change_bid
    output["ob__queue_position_change_ask"] = queue_position_change_ask
    output["ob__queue_resilience"] = queue_resilience
    output["ob__queue_imbalance_momentum"] = queue_imbalance_momentum
    output["ob__hidden_liquidity_indicator"] = hidden_liquidity_indicator.astype(bool)
    output["ob__hidden_to_visible_ratio"] = hidden_to_visible_ratio
    output["ob__liquidity_void_flag"] = liquidity_void_flag.astype(bool)
    output["ob__bid_dominance_factor"] = bid_dominance_factor
    output["ob__microstructure_pressure_velocity"] = microstructure_pressure_velocity
    output["ob__depth_elasticity"] = depth_elasticity

    float_columns = [
        "ob__total_depth_bid",
        "ob__total_depth_ask",
        "ob__top_level_size_bid",
        "ob__top_level_size_ask",
        "ob__imbalance",
        "ob__signed_book_pressure",
        "ob__imbalance_spread_adj",
        "ob__depth_imbalance_by_distance",
        "ob__level_decay_rate",
        "ob__orderbook_gap",
        "ob__bid_slope",
        "ob__ask_slope",
        "ob__curvature",
        "ob__depth_skew",
        "ob__book_entropy",
        "ob__cancellation_rate",
        "ob__new_order_rate",
        "ob__top_of_book_turnover",
        "ob__queue_position_change_bid",
        "ob__queue_position_change_ask",
        "ob__queue_resilience",
        "ob__queue_imbalance_momentum",
        "ob__hidden_to_visible_ratio",
        "ob__bid_dominance_factor",
        "ob__microstructure_pressure_velocity",
        "ob__depth_elasticity",
    ]
    for col in float_columns:
        output[col] = output[col].astype("float32")

    return output

