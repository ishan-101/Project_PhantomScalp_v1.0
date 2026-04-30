"""Smoke test for the cycle_fft_ehlers base feature family."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.cycle_fft_ehlers.features import (  # noqa: E402
    FEATURE_NAMES,
    FFT_WINDOW,
    FeatureComputationError,
    compute_cycle_fft_ehlers_features,
)
from feature_engineering.cycle_fft_ehlers.validator import (  # noqa: E402
    FeatureValidationError,
    validate_cycle_fft_ehlers_features,
)


def _build_synthetic_frame(length: int = 256) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="T")
    base_trend = np.linspace(100.0, 101.0, num=length)
    oscillation = 0.75 * np.sin(np.linspace(0, 4 * np.pi, num=length))
    return pd.DataFrame({"timestamp": timestamps, "price__mid": base_trend + oscillation})


def _assert_no_nulls(df: pd.DataFrame, columns) -> None:
    for col in columns:
        if df[col].isna().any():
            raise AssertionError(f"Nulls detected in column {col}")


def _assert_range(df: pd.DataFrame, col: str, lower: float, upper: float) -> None:
    series = df[col]
    if not series.between(lower, upper).all():
        raise AssertionError(f"Range violation in {col}: outside [{lower}, {upper}]")


def run_smoke_test() -> None:
    print("Step 1 — build synthetic input")
    base = _build_synthetic_frame()
    original_columns = list(base.columns)

    print("Step 2 — compute deterministic cycle features")
    features_df = compute_cycle_fft_ehlers_features(base)

    new_columns = [col for col in features_df.columns if col not in original_columns]
    if set(new_columns) != set(FEATURE_NAMES):
        raise AssertionError("Unexpected feature columns computed.")
    if len(new_columns) != 8:
        raise AssertionError("Feature count mismatch; expected 8 new columns.")

    print("Step 3 — validate schema and ranges")
    diagnostics = validate_cycle_fft_ehlers_features(features_df)
    if not diagnostics.get("validated"):
        raise AssertionError("Validator did not confirm successful validation.")

    _assert_no_nulls(features_df, FEATURE_NAMES)
    _assert_range(features_df, "cycle__hilbert_phase", -np.pi, np.pi)
    _assert_range(features_df, "cycle__instantaneous_frequency", -np.pi, np.pi)
    _assert_range(features_df, "cycle__phase_acceleration", -2 * np.pi, 2 * np.pi)
    _assert_range(features_df, "cycle__phase_consistency", 0.0, 1.0)
    _assert_range(features_df, "cycle__dominant_period", 1.0, float(FFT_WINDOW))

    print("Step 4 — corrupt one value to confirm validator failure")
    corrupted = features_df.copy()
    corrupted.loc[corrupted.index[0], "cycle__dominant_period"] = float(FFT_WINDOW) + 5.0
    try:
        validate_cycle_fft_ehlers_features(corrupted)
    except FeatureValidationError:
        print("PASS: validator rejected corrupted dominant period.")
    else:
        raise AssertionError("Validator did not reject corrupted dominant period.")

    print("\ncycle_fft_ehlers base features — SAFE TO FREEZE")
    print("Input → output: timestamp + mid/close price → 8 deterministic cycle features")
    print("Family status: FROZEN (no adaptive parameters, no future data)")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except (AssertionError, FeatureComputationError, FeatureValidationError) as exc:
        raise SystemExit(f"Smoke test failed: {exc}") from exc
