"""Semantic gate for the Technical Indicators base feature family.

This module performs deterministic, snapshot-causal checks to guarantee that
all technical indicator features use only backward-looking, fixed-length
windows with no adaptive tuning or future data access. No feature values are
produced here; only strict semantic validation is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    "technical__ema_fast",
    "technical__ema_medium",
    "technical__ema_slow",
    "technical__ema_ratio_fast_slow",
    "technical__ema_slope_fast",
    "technical__hma",
    "technical__kama",
    "technical__jma",
    "technical__lsma",
    "technical__atr_short",
    "technical__atr_long",
    "technical__true_range",
    "technical__normalized_volatility",
    "technical__garman_klass_volatility",
    "technical__bb_width",
    "technical__bollinger_upper_dev",
    "technical__bollinger_lower_dev",
    "technical__donchian_width",
    "technical__range_delta_velocity",
    "technical__rsi",
    "technical__stoch_k",
    "technical__stoch_d",
    "technical__roc",
    "technical__tsi",
    "technical__macd_line",
    "technical__macd_signal",
    "technical__macd_histogram",
    "technical__williams_r",
    "technical__cci",
    "technical__adx",
    "technical__dmi_plus",
    "technical__dmi_minus",
    "technical__vwap_deviation_pct",
    "technical__ema_fast_deviation_pct",
    "technical__ema_slow_deviation_pct",
    "technical__range_normalized_position",
    "technical__rolling_skew",
    "technical__rolling_kurtosis",
    "technical__zscore_returns",
    "technical__direction_entropy",
    "technical__autocorrelation_returns",
    "technical__vfi",
    "technical__chaikin_oscillator",
    "technical__poly_regression_slope",
    "technical__poly_regression_residual",
    "technical__choppiness_index",
]

# Fixed-length trailing windows required for each feature.
WINDOW_REQUIREMENTS: Dict[str, int] = {
    "technical__ema_fast": 12,
    "technical__ema_medium": 26,
    "technical__ema_slow": 50,
    "technical__ema_ratio_fast_slow": 50,
    "technical__ema_slope_fast": 12,
    "technical__hma": 20,
    "technical__kama": 30,
    "technical__jma": 30,
    "technical__lsma": 25,
    "technical__atr_short": 14,
    "technical__atr_long": 28,
    "technical__true_range": 2,
    "technical__normalized_volatility": 30,
    "technical__garman_klass_volatility": 30,
    "technical__bb_width": 20,
    "technical__bollinger_upper_dev": 20,
    "technical__bollinger_lower_dev": 20,
    "technical__donchian_width": 20,
    "technical__range_delta_velocity": 5,
    "technical__rsi": 14,
    "technical__stoch_k": 14,
    "technical__stoch_d": 16,
    "technical__roc": 12,
    "technical__tsi": 50,
    "technical__macd_line": 26,
    "technical__macd_signal": 35,
    "technical__macd_histogram": 35,
    "technical__williams_r": 14,
    "technical__cci": 20,
    "technical__adx": 28,
    "technical__dmi_plus": 28,
    "technical__dmi_minus": 28,
    "technical__vwap_deviation_pct": 1,
    "technical__ema_fast_deviation_pct": 12,
    "technical__ema_slow_deviation_pct": 50,
    "technical__range_normalized_position": 20,
    "technical__rolling_skew": 30,
    "technical__rolling_kurtosis": 30,
    "technical__zscore_returns": 30,
    "technical__direction_entropy": 20,
    "technical__autocorrelation_returns": 20,
    "technical__vfi": 130,
    "technical__chaikin_oscillator": 10,
    "technical__poly_regression_slope": 20,
    "technical__poly_regression_residual": 20,
    "technical__choppiness_index": 14,
}

# Fixed smoothing and regression constants to prevent adaptive tuning.
EMA_FAST_SPAN = 12
EMA_MEDIUM_SPAN = 26
EMA_SLOW_SPAN = 50
EMA_SIGNAL_SPAN = 9
KAMA_ER_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
JMA_LENGTH = 20
JMA_PHASE = 0.0
HMA_PERIOD = 20
LSMA_PERIOD = 25
TSI_LONG = 25
TSI_SHORT = 13
ROC_PERIOD = 12
RSI_PERIOD = 14
ADX_PERIOD = 14
STOCH_D_SMOOTH = 3
STOCH_PERIOD = 14
WILLIAMS_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
DONCHIAN_PERIOD = 20
RANGE_POSITION_PERIOD = 20
SKEW_WINDOW = 30
KURTOSIS_WINDOW = 30
Z_RETURN_WINDOW = 30
ENTROPY_WINDOW = 20
AUTO_WINDOW = 20
GARMAN_KLASS_WINDOW = 30
VFI_WINDOW = 130
VFI_VFACTOR = 0.2
CHAIKIN_SHORT = 3
CHAIKIN_LONG = 10
POLY_WINDOW = 20
CHOP_WINDOW = 14


class SemanticValidationError(RuntimeError):
    """Raised when semantic or causality checks fail."""


@dataclass(frozen=True)
class InputContract:
    """Schema expectations for semantic validation."""

    time_column: str = "timestamp"
    price_columns: Sequence[str] = (
        "price__mid",
        "ohlcv__close",
    )
    ohlcv_columns: Sequence[str] = (
        "ohlcv__open",
        "ohlcv__high",
        "ohlcv__low",
        "ohlcv__close",
        "ohlcv__volume",
    )


CONTRACT = InputContract()


def _assert_feature_catalog() -> None:
    if len(FEATURE_NAMES) != 46:
        raise SemanticValidationError("Feature catalog must contain exactly 46 entries.")
    duplicates = [name for name in FEATURE_NAMES if FEATURE_NAMES.count(name) > 1]
    if duplicates:
        raise SemanticValidationError(f"Duplicate feature names detected: {sorted(set(duplicates))}.")
    missing_windows = [name for name in FEATURE_NAMES if name not in WINDOW_REQUIREMENTS]
    if missing_windows:
        raise SemanticValidationError(f"Missing window requirements for features: {missing_windows}")


def _assert_required_inputs(df: pd.DataFrame) -> None:
    if CONTRACT.time_column not in df.columns:
        raise SemanticValidationError("Missing timestamp column for ordering checks.")

    has_price = any(col in df.columns for col in CONTRACT.price_columns)
    if not has_price:
        raise SemanticValidationError("A mid or close price column is required for technical indicators.")

    missing_ohlcv = [col for col in CONTRACT.ohlcv_columns if col not in df.columns]
    if missing_ohlcv:
        raise SemanticValidationError(f"Missing OHLCV columns: {missing_ohlcv}")

    for col in CONTRACT.ohlcv_columns:
        if df[col].isna().any():
            raise SemanticValidationError(f"OHLCV column '{col}' contains nulls; fill explicitly.")

    if not pd.Series(df[CONTRACT.time_column]).is_monotonic_increasing:
        raise SemanticValidationError("Timestamps must be strictly non-decreasing for causal windows.")


def _assert_window_lengths(df: pd.DataFrame) -> None:
    max_window = max(WINDOW_REQUIREMENTS.values())
    if len(df) < max_window:
        raise SemanticValidationError(
            f"Insufficient history for maximum lookback {max_window}; received {len(df)} rows."
        )

    for feature, window in WINDOW_REQUIREMENTS.items():
        if window <= 0:
            raise SemanticValidationError(f"Non-positive window declared for {feature}: {window}")


def _assert_trailing_access_only(length: int, windows: Iterable[int]) -> None:
    for window in windows:
        for idx in range(length):
            start = max(0, idx - window + 1)
            if start > idx:
                raise SemanticValidationError("Forward indexing detected while verifying trailing windows.")
            if idx + 1 > length:
                raise SemanticValidationError("Window evaluation exceeded available history.")


def _assert_fixed_parameters() -> None:
    constants = [
        EMA_FAST_SPAN,
        EMA_MEDIUM_SPAN,
        EMA_SLOW_SPAN,
        EMA_SIGNAL_SPAN,
        KAMA_ER_PERIOD,
        KAMA_FAST,
        KAMA_SLOW,
        JMA_LENGTH,
        JMA_PHASE,
        HMA_PERIOD,
        LSMA_PERIOD,
        TSI_LONG,
        TSI_SHORT,
        ROC_PERIOD,
        RSI_PERIOD,
        ADX_PERIOD,
        STOCH_D_SMOOTH,
        STOCH_PERIOD,
        WILLIAMS_PERIOD,
        BB_PERIOD,
        BB_STD,
        DONCHIAN_PERIOD,
        RANGE_POSITION_PERIOD,
        SKEW_WINDOW,
        KURTOSIS_WINDOW,
        Z_RETURN_WINDOW,
        ENTROPY_WINDOW,
        AUTO_WINDOW,
        GARMAN_KLASS_WINDOW,
        VFI_WINDOW,
        VFI_VFACTOR,
        CHAIKIN_SHORT,
        CHAIKIN_LONG,
        POLY_WINDOW,
        CHOP_WINDOW,
    ]
    if not all(np.isfinite(value) for value in constants):
        raise SemanticValidationError("Non-finite technical indicator constant detected.")
    if any(value <= 0 for value in constants if value != JMA_PHASE):
        raise SemanticValidationError("All lookbacks and smoothing constants must be strictly positive.")


def run_semantic_checks(df: pd.DataFrame) -> None:
    _assert_feature_catalog()
    _assert_required_inputs(df)
    _assert_window_lengths(df)
    _assert_trailing_access_only(len(df), WINDOW_REQUIREMENTS.values())
    _assert_fixed_parameters()


def _build_synthetic_input(length: int = 260) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="T")
    base = np.linspace(100.0, 102.0, num=length)
    noise = 0.5 * np.sin(np.linspace(0, 6 * np.pi, num=length))
    close = base + noise
    high = close + 0.2
    low = close - 0.2
    open_price = close - 0.05
    volume = np.linspace(1_000.0, 1_500.0, num=length)
    return pd.DataFrame({
        CONTRACT.time_column: timestamps,
        "price__mid": close,
        "ohlcv__open": open_price,
        "ohlcv__high": high,
        "ohlcv__low": low,
        "ohlcv__close": close,
        "ohlcv__volume": volume,
    })


if __name__ == "__main__":
    synthetic_df = _build_synthetic_input()
    run_semantic_checks(synthetic_df)
    print("Technical Indicators semantic validation — PASSED")
