"""Deterministic computation of Higher-Order engineered features.

All computations:
- rely solely on frozen base features plus explicit immutable constants,
- avoid any lookahead, labeling logic, or model fitting,
- are snapshot-causal and auditable,
- reject nulls rather than silently coercing or filling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when higher-order feature computation cannot proceed."""


@dataclass(frozen=True)
class HigherOrderConfig:
    """Immutable configuration for higher-order feature computation."""

    baseline_price_mean: float
    baseline_price_std: float
    baseline_volume_mean: float
    baseline_volume_std: float
    baseline_ob_imbalance_mean: float
    baseline_ob_imbalance_std: float
    pca_book_loadings: Sequence[float]
    ehlers_coefficients: Sequence[float]
    signed_volume_alpha: float


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required base feature columns: {missing}")


def _assert_no_nulls(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if df[col].isna().any():
            raise FeatureComputationError(f"Null values detected in required column '{col}'.")


def _validate_config(config: HigherOrderConfig) -> None:
    if config.baseline_price_std <= 0:
        raise FeatureComputationError("baseline_price_std must be positive.")
    if config.baseline_volume_std <= 0:
        raise FeatureComputationError("baseline_volume_std must be positive.")
    if config.baseline_ob_imbalance_std <= 0:
        raise FeatureComputationError("baseline_ob_imbalance_std must be positive.")
    if len(config.pca_book_loadings) != 4:
        raise FeatureComputationError("pca_book_loadings must contain exactly 4 loadings.")
    if len(config.ehlers_coefficients) != 5:
        raise FeatureComputationError("ehlers_coefficients must contain [b0, b1, b2, a1, a2].")
    if not (0 < config.signed_volume_alpha <= 1):
        raise FeatureComputationError("signed_volume_alpha must be in (0, 1].")


def _z_score(series: pd.Series, mean: float, std: float) -> pd.Series:
    return ((series.astype(float) - mean) / std).astype("float32")


def _pca_component(size_vector: pd.DataFrame, loadings: Sequence[float]) -> pd.Series:
    matrix = size_vector.to_numpy(dtype=float)
    loading_array = np.asarray(loadings, dtype=float)
    component = matrix @ loading_array
    return pd.Series(component, index=size_vector.index, dtype="float32")


def _ehlers_filter(series: pd.Series, coefficients: Sequence[float]) -> pd.Series:
    """Second-order recursive filter using fixed coefficients (b0, b1, b2, a1, a2)."""

    b0, b1, b2, a1, a2 = (float(x) for x in coefficients)
    values = series.astype(float).to_numpy()
    output = np.zeros_like(values, dtype=float)

    for idx, value in enumerate(values):
        x0 = value
        x1 = values[idx - 1] if idx - 1 >= 0 else value
        x2 = values[idx - 2] if idx - 2 >= 0 else x1
        y1 = output[idx - 1] if idx - 1 >= 0 else value
        y2 = output[idx - 2] if idx - 2 >= 0 else y1
        output[idx] = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

    return pd.Series(output, index=series.index, dtype="float32")


def _ema(series: pd.Series, alpha: float) -> pd.Series:
    values = series.astype(float).to_numpy()
    ema_values = np.zeros_like(values, dtype=float)
    for idx, value in enumerate(values):
        if idx == 0:
            ema_values[idx] = value
        else:
            ema_values[idx] = alpha * value + (1 - alpha) * ema_values[idx - 1]
    return pd.Series(ema_values, index=series.index, dtype="float32")


def compute_higher_order_features(df: pd.DataFrame, *, config: HigherOrderConfig) -> pd.DataFrame:
    """Compute the 14 Higher-Order engineered features deterministically."""

    _validate_config(config)

    required_columns = [
        "price__last",
        "price__mid",
        "volume__tick",
        "spread__l1",
        "ob__imbalance",
        "ob__top_level_size_bid",
        "ob__top_level_size_ask",
        "ob__total_depth_bid",
        "ob__total_depth_ask",
        "tick_return",
        "of__signed_volume",
        "of__price_impact_per_unit_volume",
        "regime__liquidity_score",
    ]
    _require_columns(df, required_columns)
    _assert_no_nulls(df, required_columns)

    working = df.copy()

    # 1) Z-score of price.
    working["ho__z_price"] = _z_score(working["price__last"], config.baseline_price_mean, config.baseline_price_std)

    # 2) Log return using tick return to avoid lookahead.
    tick_return = working["tick_return"].astype(float)
    if (tick_return <= -1).any():
        raise FeatureComputationError("tick_return must remain greater than -1 to compute log returns.")
    log_return = np.log1p(tick_return)
    working["ho__log_return"] = log_return.astype("float32")

    # 3) Second-derivative proxy of price returns.
    price_accel = log_return.diff().fillna(0.0)
    working["ho__price_accel"] = price_accel.astype("float32")

    # 4) Volume z-score.
    working["ho__volume_z"] = _z_score(working["volume__tick"], config.baseline_volume_mean, config.baseline_volume_std)

    # 5) Order-book imbalance z-score.
    working["ho__ob_imbalance_z"] = _z_score(
        working["ob__imbalance"], config.baseline_ob_imbalance_mean, config.baseline_ob_imbalance_std
    )

    # 6) First PCA component of book depth vector.
    depth_vector = working[
        ["ob__top_level_size_bid", "ob__top_level_size_ask", "ob__total_depth_bid", "ob__total_depth_ask"]
    ]
    working["ho__pca1_book"] = _pca_component(depth_vector, config.pca_book_loadings)

    # 7) Interaction: volume × spread.
    working["ho__interaction_of_volume_spread"] = (
        working["volume__tick"].astype(float) * working["spread__l1"].astype(float)
    ).astype("float32")

    # 8) Return × signed volume interaction.
    working["ho__return_signed_volume_interaction"] = (
        working["tick_return"].astype(float) * working["of__signed_volume"].astype(float)
    ).astype("float32")

    # 9) Return normalized by spread (strictly require positive spread).
    spread = working["spread__l1"].astype(float)
    if (spread <= 0).any():
        raise FeatureComputationError("spread__l1 must be positive to compute return_over_spread.")
    working["ho__return_over_spread"] = safe_divide(
        working["tick_return"].astype(float),
        spread,
        allow_zero_division=False,
    ).astype("float32")

    # 10) Imbalance × return.
    working["ho__imbalance_times_return"] = (
        working["ob__imbalance"].astype(float) * working["tick_return"].astype(float)
    ).astype("float32")

    # 11) Ehlers-filtered mid price using fixed coefficients.
    working["ho__filtered_mid_price"] = _ehlers_filter(working["price__mid"], config.ehlers_coefficients)

    # 12) Order-book imbalance × signed orderflow.
    working["ho__book_flow_interaction"] = (
        working["ob__imbalance"].astype(float) * working["of__signed_volume"].astype(float)
    ).astype("float32")

    # 13) Impact per unit volume normalized by market regime.
    liquidity_score = working["regime__liquidity_score"].astype(float)
    if (liquidity_score <= 0).any():
        raise FeatureComputationError("regime__liquidity_score must be strictly positive for normalization.")
    working["ho__normalized_impact"] = safe_divide(
        working["of__price_impact_per_unit_volume"].astype(float),
        liquidity_score,
        allow_zero_division=False,
    ).astype("float32")

    # 14) EMA of signed volume with fixed alpha.
    working["ho__signed_volume_ema"] = _ema(working["of__signed_volume"], config.signed_volume_alpha)

    output_columns = [
        "ho__z_price",
        "ho__log_return",
        "ho__price_accel",
        "ho__volume_z",
        "ho__ob_imbalance_z",
        "ho__pca1_book",
        "ho__interaction_of_volume_spread",
        "ho__return_signed_volume_interaction",
        "ho__return_over_spread",
        "ho__imbalance_times_return",
        "ho__filtered_mid_price",
        "ho__book_flow_interaction",
        "ho__normalized_impact",
        "ho__signed_volume_ema",
    ]

    return working[output_columns]
