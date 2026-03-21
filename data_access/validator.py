"""Validation gate for raw data prior to feature engineering."""

from __future__ import annotations

from typing import List

import pandas as pd
from pandas.api import types as ptypes

from .raw_schema import FeedSchema


class ValidationError(ValueError):
    """Raised when data fails structural validation."""


def _assert_columns_present(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")


def _assert_no_extra_columns(df: pd.DataFrame, schema: FeedSchema) -> None:
    allowed = {field.name for field in schema.fields}
    extras = [col for col in df.columns if col not in allowed]
    if extras:
        raise ValidationError(
            "Unexpected columns present (not in schema): " + ", ".join(extras)
        )


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "datetime64[ns]":
        return ptypes.is_datetime64_ns_dtype(series) or ptypes.is_datetime64tz_dtype(series)
    if expected == "string":
        return ptypes.is_string_dtype(series)
    if expected == "float64":
        return ptypes.is_float_dtype(series) and series.dtype == "float64"
    if expected == "int64":
        return ptypes.is_integer_dtype(series) and series.dtype == "int64"
    return str(series.dtype) == expected


def _assert_dtypes(df: pd.DataFrame, schema: FeedSchema) -> None:
    mismatches = []
    for field in schema.fields:
        if field.name not in df.columns:
            if field.required:
                mismatches.append(f"{field.name} (missing)")
            continue
        if not _dtype_matches(df[field.name], field.dtype):
            mismatches.append(f"{field.name} expected {field.dtype} got {df[field.name].dtype}")
    if mismatches:
        raise ValidationError("Dtype mismatches: " + "; ".join(mismatches))


def _assert_no_nulls(df: pd.DataFrame, schema: FeedSchema) -> None:
    nulls = []
    for field in schema.fields:
        if not field.required or field.name not in df.columns:
            continue
        if df[field.name].isnull().any():
            nulls.append(field.name)
    if nulls:
        raise ValidationError("Nulls detected in required columns: " + ", ".join(nulls))


def _assert_timestamp_sanity(df: pd.DataFrame, schema: FeedSchema) -> None:
    ts_col = schema.timestamp_column
    if ts_col not in df.columns:
        raise ValidationError(f"Timestamp column '{ts_col}' missing.")
    ts_series = df[ts_col]
    if not _dtype_matches(ts_series, "datetime64[ns]"):
        raise ValidationError(
            f"Timestamp column '{ts_col}' must be datetime64[ns] (tz-aware allowed)."
        )
    if ts_series.isnull().any():
        raise ValidationError(f"Timestamp column '{ts_col}' contains null/NaT values.")


def _assert_no_exact_duplicates(df: pd.DataFrame, schema: FeedSchema) -> None:
    if schema.allow_exact_duplicates:
        return
    if df.duplicated().any():
        raise ValidationError(
            f"Exact duplicate rows detected in feed '{schema.feed_name}' and duplicates are not allowed."
        )


def validate(df: pd.DataFrame, schema: FeedSchema) -> None:
    """Run structural validations against a DataFrame.

    This function performs no mutation. Any violation raises ValidationError with
    a human-readable message to enforce fail-fast behavior.
    """

    _assert_columns_present(df, [f.name for f in schema.required_fields])
    _assert_no_extra_columns(df, schema)
    _assert_dtypes(df, schema)
    _assert_no_nulls(df, schema)
    _assert_timestamp_sanity(df, schema)
    _assert_no_exact_duplicates(df, schema)
