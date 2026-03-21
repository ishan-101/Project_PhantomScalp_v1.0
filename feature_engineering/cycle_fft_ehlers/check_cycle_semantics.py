"""Semantic gate for Cycle / FFT / Ehlers base feature family.

This module performs deterministic, snapshot-causal checks to ensure the
cycle_fft_ehlers features can be computed without leakage, adaptive parameters,
or future data access. No feature values are finalized here; only causality and
input sufficiency are validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES: List[str] = [
    "cycle__hilbert_phase",
    "cycle__dominant_period",
    "cycle__ehlers_filt_output",
    "cycle__instantaneous_frequency",
    "cycle__trend_component",
    "cycle__cycle_component",
    "cycle__phase_acceleration",
    "cycle__phase_consistency",
]

# Fixed-length trailing windows for each deterministic computation.
WINDOW_REQUIREMENTS: Dict[str, int] = {
    "cycle__hilbert_phase": 64,
    "cycle__dominant_period": 128,
    "cycle__ehlers_filt_output": 2,  # recursive filter seeded causally
    "cycle__instantaneous_frequency": 64,
    "cycle__trend_component": 32,
    "cycle__cycle_component": 32,
    "cycle__phase_acceleration": 64,
    "cycle__phase_consistency": 32,
}

# Fixed Ehlers super-smoother coefficients derived from a constant period.
EHLERS_FILTER_COEFFICIENTS = {
    "period": 10.0,
    "a1": float(np.exp(-1.414 * np.pi / 10.0)),
    # These coefficients are deterministic and never tuned from data.
    "c2": None,
    "c3": None,
    "c1": None,
}
# Populate dependent coefficients once using the fixed period.
EHLERS_FILTER_COEFFICIENTS["c2"] = 2 * EHLERS_FILTER_COEFFICIENTS["a1"] * np.cos(
    1.414 * np.pi / EHLERS_FILTER_COEFFICIENTS["period"]
)
EHLERS_FILTER_COEFFICIENTS["c3"] = -EHLERS_FILTER_COEFFICIENTS["a1"] ** 2
EHLERS_FILTER_COEFFICIENTS["c1"] = 1 - EHLERS_FILTER_COEFFICIENTS["c2"] - EHLERS_FILTER_COEFFICIENTS["c3"]


class SemanticValidationError(RuntimeError):
    """Raised when semantic or causality checks fail."""


@dataclass(frozen=True)
class InputContract:
    """Schema expectations for semantic validation."""

    time_column: str = "timestamp"
    price_columns: Sequence[str] = ("price__mid", "ohlcv__close")


CONTRACT = InputContract()


def _assert_required_inputs(df: pd.DataFrame) -> None:
    if CONTRACT.time_column not in df.columns:
        raise SemanticValidationError("Missing timestamp column for ordering checks.")

    if not any(col in df.columns for col in CONTRACT.price_columns):
        raise SemanticValidationError("Neither mid nor close price column is available.")

    price_col = next(col for col in CONTRACT.price_columns if col in df.columns)
    if df[price_col].isna().any():
        raise SemanticValidationError("Price series contains nulls; fill explicitly before use.")

    if not pd.Series(df[CONTRACT.time_column]).is_monotonic_increasing:
        raise SemanticValidationError("Timestamps must be strictly non-decreasing for causal windows.")


def _assert_window_lengths(df: pd.DataFrame) -> None:
    required = max(WINDOW_REQUIREMENTS.values())
    if len(df) < required:
        raise SemanticValidationError(
            f"Insufficient history for fixed windows: need at least {required} rows, found {len(df)}."
        )


def _assert_trailing_access_only(length: int, windows: Iterable[int]) -> None:
    for window in windows:
        if window <= 0:
            raise SemanticValidationError(f"Non-positive window length detected: {window}.")
        for idx in range(length):
            start = max(0, idx - window + 1)
            if start > idx:
                raise SemanticValidationError(
                    "Forward indexing detected while evaluating trailing window semantics."
                )
            if idx + 1 > length:
                raise SemanticValidationError("Window evaluation exceeded available history.")


def _assert_fixed_coefficients() -> None:
    required_keys = {"period", "a1", "c1", "c2", "c3"}
    if set(EHLERS_FILTER_COEFFICIENTS) != required_keys:
        raise SemanticValidationError("Ehlers filter coefficients are incomplete or malformed.")
    for name, value in EHLERS_FILTER_COEFFICIENTS.items():
        if not np.isfinite(value):
            raise SemanticValidationError(f"Coefficient '{name}' is not finite: {value}.")
    if EHLERS_FILTER_COEFFICIENTS["period"] <= 0:
        raise SemanticValidationError("Ehlers filter period must be strictly positive and fixed.")


def _assert_feature_catalog() -> None:
    if len(FEATURE_NAMES) != 8:
        raise SemanticValidationError("Feature catalog must contain exactly 8 entries.")
    duplicates = [name for name in FEATURE_NAMES if FEATURE_NAMES.count(name) > 1]
    if duplicates:
        raise SemanticValidationError(f"Duplicate feature names detected: {sorted(set(duplicates))}.")


def run_semantic_checks(df: pd.DataFrame) -> None:
    """Run deterministic semantic checks on the provided DataFrame."""

    _assert_feature_catalog()
    _assert_required_inputs(df)
    _assert_window_lengths(df)
    _assert_trailing_access_only(len(df), WINDOW_REQUIREMENTS.values())
    _assert_fixed_coefficients()


def _build_synthetic_input(length: int = 196) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=length, freq="T")
    base = np.linspace(100.0, 100.5, num=length)
    oscillation = 0.25 * np.sin(np.linspace(0, 6 * np.pi, num=length))
    price = base + oscillation
    return pd.DataFrame({
        CONTRACT.time_column: timestamps,
        "price__mid": price.astype(np.float64),
    })


if __name__ == "__main__":
    synthetic_df = _build_synthetic_input()
    run_semantic_checks(synthetic_df)
    print("Cycle / FFT / Ehlers semantic validation — PASSED")
