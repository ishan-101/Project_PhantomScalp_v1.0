"""Per-record computation of orderflow_tick base features using current and previous data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when feature computation cannot proceed due to invalid inputs."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _as_array(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple, pd.Series)):
        return np.asarray(value, dtype=float)
    try:
        return np.array([float(value)], dtype=float)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise FeatureComputationError(f"Cannot coerce value to array: {value}") from exc


def _as_int_array(value: object) -> np.ndarray:
    arr = _as_array(value)
    return arr.astype(int)


def _mean_or_zero(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else 0.0


def _sum_or_zero(arr: np.ndarray) -> float:
    return float(arr.sum()) if arr.size else 0.0


def compute_orderflow_tick_features(
    df: pd.DataFrame,
    *,
    small_trade_threshold: float,
    large_trade_threshold: float,
    decay_lambda: float = 0.5,
) -> pd.DataFrame:
    """Compute orderflow_tick base features without lookahead or rolling state."""

    if small_trade_threshold < 0:
        raise FeatureComputationError("small_trade_threshold must be non-negative.")
    if large_trade_threshold <= small_trade_threshold:
        raise FeatureComputationError("large_trade_threshold must exceed small_trade_threshold.")
    if decay_lambda <= 0 or decay_lambda > 1:
        raise FeatureComputationError("decay_lambda must be in (0, 1].")

    required = [
        "trade_price",
        "trade_size",
        "aggressor_side",
        "trade_count",
        "timestamp",
        "mid_price",
        "spread",
        "visible_depth",
        "vwap",
        "previous_trade_price",
        "previous_aggressor_side",
        "previous_signed_volume",
    ]
    _require_columns(df, required)

    output = df.copy()

    arrays_price = df["trade_price"].apply(_as_array)
    arrays_size = df["trade_size"].apply(_as_array)
    arrays_aggr = df["aggressor_side"].apply(_as_int_array)

    # Totals and directional splits.
    signed_volume = arrays_size.combine(arrays_aggr, lambda s, a: _sum_or_zero(s * a))
    buy_volume = arrays_size.combine(arrays_aggr, lambda s, a: _sum_or_zero(s[a > 0]))
    sell_volume = arrays_size.combine(arrays_aggr, lambda s, a: _sum_or_zero(s[a < 0]))
    total_volume = arrays_size.apply(_sum_or_zero)

    output["of__signed_volume"] = signed_volume.astype(float)
    output["of__aggressive_buy_size"] = buy_volume.astype(float)
    output["of__aggressive_sell_size"] = sell_volume.astype(float)

    denom_volume = buy_volume + sell_volume
    imbalance = safe_divide(
        buy_volume - sell_volume,
        denom_volume.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)
    output["of__imbalance_ratio"] = imbalance

    output["of__large_trade_count"] = arrays_size.apply(lambda arr: int((arr > large_trade_threshold).sum())).astype(
        "int32"
    )

    trade_count = output["trade_count"].fillna(0).astype("int32")
    output["trade_count"] = trade_count

    output["of__avg_trade_size"] = safe_divide(
        total_volume,
        trade_count.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    timestamps = df["timestamp"]
    def _avg_gap(ts_value: object) -> float:
        ts_array = _as_array(ts_value)
        if ts_array.size <= 1:
            return 0.0
        diffs = np.diff(ts_array)
        return float(np.mean(diffs)) if diffs.size else 0.0

    output["of__time_between_trades"] = timestamps.apply(_avg_gap)

    output["of__aggressor_flag_ratio"] = safe_divide(
        arrays_aggr.apply(lambda arr: float((arr > 0).sum())),
        trade_count.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    mean_price = arrays_price.apply(_mean_or_zero)
    vwap = output["vwap"].fillna(mean_price)
    output["vwap"] = vwap
    output["of__trade_price_vs_vwap"] = safe_divide(
        mean_price - vwap,
        vwap.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    prev_trade_price = _as_array(df["previous_trade_price"]).astype(float)
    prev_trade_price_series = pd.Series(prev_trade_price[: len(df)], index=df.index)

    def _run_lengths(prices: np.ndarray, prev_price: float) -> tuple[int, int]:
        if prices.size == 0:
            return 0, 0
        last_price = prev_price
        up = 0
        down = 0
        for price in prices:
            if price > last_price:
                up += 1
                down = 0
            elif price < last_price:
                down += 1
                up = 0
            else:
                up = 0
                down = 0
            last_price = price
        return up, down

    run_up: list[int] = []
    run_down: list[int] = []
    for idx, prices in arrays_price.items():
        prev_price = prev_trade_price[idx] if idx < prev_trade_price.size else 0.0
        up, down = _run_lengths(prices, prev_price)
        run_up.append(up)
        run_down.append(down)

    output["of__run_length_up"] = pd.Series(run_up, index=df.index, dtype="int32")
    output["of__run_length_down"] = pd.Series(run_down, index=df.index, dtype="int32")

    output["of__sequence_entropy"] = arrays_aggr.apply(
        lambda arr: 0.0
        if arr.size == 0
        else float(
            -np.sum(
                [
                    p * np.log(p)
                    for p in [np.mean(arr > 0), np.mean(arr < 0)]
                    if p > 0
                ]
            )
        )
    )

    output["of__small_trade_vol"] = arrays_size.apply(lambda arr: _sum_or_zero(arr[arr < small_trade_threshold]))
    output["of__medium_trade_vol"] = arrays_size.apply(
        lambda arr: _sum_or_zero(arr[(arr >= small_trade_threshold) & (arr < large_trade_threshold)])
    )
    output["of__large_trade_vol"] = arrays_size.apply(lambda arr: _sum_or_zero(arr[arr >= large_trade_threshold]))

    mid_price = output["mid_price"].fillna(0.0)
    output["of__vwap_pressure"] = (vwap - mid_price).fillna(0.0)

    output["of__aggressor_volume_ratio"] = safe_divide(
        buy_volume,
        sell_volume.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["of__execution_flow_polarity"] = np.sign(signed_volume).astype("int32")

    visible_depth = output["visible_depth"].replace(0, np.nan)
    output["of__market_pressure_tilt"] = safe_divide(
        signed_volume * output["spread"],
        visible_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    price_impact = safe_divide(
        mean_price - mid_price,
        output["spread"].replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)
    output["of__impact_adjusted_flow"] = (signed_volume * price_impact).astype(float)

    prev_signed_volume = output["previous_signed_volume"].fillna(0.0)
    prev_sign = np.sign(prev_signed_volume)
    current_sign = np.sign(signed_volume)
    output["of__aggression_persistence"] = (current_sign != 0) & (current_sign == prev_sign)

    def _elapsed_seconds(ts_value: object) -> float:
        ts_array = _as_array(ts_value)
        if ts_array.size <= 1:
            return 1.0
        return float(np.max(ts_array) - np.min(ts_array))

    elapsed = timestamps.apply(_elapsed_seconds)
    output["of__trade_burst_intensity"] = safe_divide(
        trade_count.astype(float),
        elapsed.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["of__toxicity_proxy"] = safe_divide(
        signed_volume * (mean_price - mid_price),
        visible_depth,
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    price_direction = np.sign(mean_price - prev_trade_price_series)
    aligned_counts = arrays_aggr.combine(price_direction, lambda arr, direction: float(
        ((arr > 0) & (direction > 0)).sum() + ((arr < 0) & (direction < 0)).sum()
    ))
    output["of__realized_sign_rate"] = safe_divide(
        aligned_counts,
        trade_count.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["of__price_impact_per_unit_volume"] = safe_divide(
        (mean_price - prev_trade_price_series).abs(),
        total_volume.replace(0, np.nan),
        allow_zero_division=True,
        fill_value=0.0,
    ).fillna(0.0)

    output["of__initiator_persistence"] = ((current_sign != 0) & (current_sign == prev_sign)).astype(float)

    output["of__time_decay_of_flow"] = (prev_signed_volume * decay_lambda + signed_volume * (1 - decay_lambda)).astype(
        float
    )

    float_features = [
        "of__signed_volume",
        "of__imbalance_ratio",
        "of__avg_trade_size",
        "of__time_between_trades",
        "of__aggressor_flag_ratio",
        "of__trade_price_vs_vwap",
        "of__aggressive_buy_size",
        "of__aggressive_sell_size",
        "of__sequence_entropy",
        "of__small_trade_vol",
        "of__medium_trade_vol",
        "of__large_trade_vol",
        "of__vwap_pressure",
        "of__aggressor_volume_ratio",
        "of__market_pressure_tilt",
        "of__impact_adjusted_flow",
        "of__trade_burst_intensity",
        "of__toxicity_proxy",
        "of__realized_sign_rate",
        "of__price_impact_per_unit_volume",
        "of__initiator_persistence",
        "of__time_decay_of_flow",
    ]
    output[float_features] = output[float_features].astype("float32")

    return output
