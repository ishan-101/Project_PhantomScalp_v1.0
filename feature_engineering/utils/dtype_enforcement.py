"""Strict dtype enforcement utilities for feature engineering infrastructure."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


class DtypeValidationError(TypeError):
    """Raised when an input DataFrame fails dtype validation."""


def _normalize_dtype(dtype: str) -> str:
    """Normalize dtype strings for consistent comparison."""
    return dtype.lower().replace(" ", "")


def validate_dtypes(df: pd.DataFrame, expected_schema: Mapping[str, str]) -> None:
    """Validate that DataFrame columns match the expected dtypes exactly.

    Args:
        df: Input DataFrame to validate.
        expected_schema: Mapping of column name to expected pandas-compatible dtype string.

    Raises:
        DtypeValidationError: If any column is missing or of the wrong dtype.
    """
    missing = [col for col in expected_schema if col not in df.columns]
    if missing:
        raise DtypeValidationError(f"Missing columns for dtype validation: {missing}")

    mismatched = {}
    for col, expected_dtype in expected_schema.items():
        actual = _normalize_dtype(str(df[col].dtype))
        expected = _normalize_dtype(expected_dtype)
        if actual != expected:
            mismatched[col] = {"expected": expected_dtype, "actual": str(df[col].dtype)}

    if mismatched:
        raise DtypeValidationError(
            "Column dtype mismatches detected: " + ", ".join(
                f"{col} (expected {info['expected']}, actual {info['actual']})"
                for col, info in mismatched.items()
            )
        )


def enforce_dtypes(df: pd.DataFrame, expected_schema: Mapping[str, str]) -> pd.DataFrame:
    """Return a copy of ``df`` with columns cast to expected dtypes when valid.

    No columns are coerced silently. Casting errors propagate as exceptions.

    Args:
        df: Input DataFrame to cast.
        expected_schema: Mapping of column name to dtype strings to enforce.

    Returns:
        A new DataFrame with columns cast to the exact expected dtypes.

    Raises:
        DtypeValidationError: If validation fails before or after casting.
        KeyError: If expected columns are missing.
    """
    missing = [col for col in expected_schema if col not in df.columns]
    if missing:
        raise DtypeValidationError(f"Cannot enforce dtypes; missing columns: {missing}")

    cast_df = df.copy()
    for col, dtype in expected_schema.items():
        try:
            cast_df[col] = cast_df[col].astype(dtype, errors="raise")
        except Exception as exc:  # pylint: disable=broad-except
            raise DtypeValidationError(
                f"Failed to cast column '{col}' to dtype '{dtype}': {exc}"
            ) from exc

    validate_dtypes(cast_df, expected_schema)
    return cast_df
