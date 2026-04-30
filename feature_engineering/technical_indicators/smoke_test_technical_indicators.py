"""Smoke test for the technical_indicators base feature family."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.technical_indicators.features import (  # noqa: E402
    FEATURE_NAMES,
    FeatureComputationError,
    compute_technical_indicator_features,
)
from feature_engineering.technical_indicators.validator import (  # noqa: E402
    FeatureValidationError,
    validate_technical_indicator_features,
)


def _build_synthetic_frame(length: int = 260) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="T")
    base_trend = np.linspace(100.0, 102.0, num=length)
    oscillation = 0.75 * np.sin(np.linspace(0, 6 * np.pi, num=length))
    close = base_trend + oscillation
    high = close + 0.25
    low = close - 0.25
    open_price = close - 0.05
    volume = np.linspace(1_000.0, 1_600.0, num=length)
    return pd.DataFrame({
        "timestamp": timestamps,
        "price__mid": close,
        "ohlcv__open": open_price,
        "ohlcv__high": high,
        "ohlcv__low": low,
        "ohlcv__close": close,
        "ohlcv__volume": volume,
    })


def _assert_no_nulls(df: pd.DataFrame, columns) -> None:
    for col in columns:
        if df[col].isna().any():
            raise AssertionError(f"Nulls detected in column {col}")


def _assert_range(df: pd.DataFrame, col: str, lower: float, upper: float) -> None:
    if not df[col].between(lower, upper).all():
        raise AssertionError(f"Range violation in {col}: outside [{lower}, {upper}]")


def run_smoke_test() -> None:
    print("Step 1 — build synthetic input")
    base = _build_synthetic_frame()
    original_columns = list(base.columns)

    print("Step 2 — compute deterministic technical indicators")
    features_df = compute_technical_indicator_features(base)

    new_columns = [col for col in features_df.columns if col not in original_columns]
    if set(new_columns) != set(FEATURE_NAMES):
        raise AssertionError("Unexpected feature columns computed.")
    if len(new_columns) != 46:
        raise AssertionError("Feature count mismatch; expected 46 new columns.")

    print("Step 3 — validate schema and ranges")
    diagnostics = validate_technical_indicator_features(features_df)
    if not diagnostics.get("validated"):
        raise AssertionError("Validator did not confirm successful validation.")

    _assert_no_nulls(features_df, FEATURE_NAMES)
    _assert_range(features_df, "technical__rsi", 0.0, 100.0)
    _assert_range(features_df, "technical__williams_r", -100.0, 0.0)
    _assert_range(features_df, "technical__adx", 0.0, 100.0)
    _assert_range(features_df, "technical__stoch_k", 0.0, 100.0)
    _assert_range(features_df, "technical__stoch_d", 0.0, 100.0)
    _assert_range(features_df, "technical__bollinger_upper_dev", -10.0, 10.0)
    _assert_range(features_df, "technical__bollinger_lower_dev", -10.0, 10.0)

    print("Step 4 — corrupt one value to confirm validator failure")
    corrupted = features_df.copy()
    corrupted.loc[corrupted.index[0], "technical__rsi"] = 150.0
    try:
        validate_technical_indicator_features(corrupted)
    except FeatureValidationError:
        print("PASS: validator rejected corrupted RSI.")
    else:
        raise AssertionError("Validator did not reject corrupted RSI.")

    print("\ntechnical_indicators base features — SAFE TO FREEZE")
    print("Input → output: timestamp + OHLCV/volume → 46 deterministic technical indicators")
    print("Family status: FROZEN (no adaptive parameters, no future data)")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except (AssertionError, FeatureComputationError, FeatureValidationError) as exc:
        raise SystemExit(f"Smoke test failed: {exc}") from exc
