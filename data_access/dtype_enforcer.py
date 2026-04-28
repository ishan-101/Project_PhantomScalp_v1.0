"""Strict dtype enforcement based on raw schemas."""

from __future__ import annotations

import pandas as pd

from .raw_schema import FieldSpec, FeedSchema


class DtypeEnforcementError(ValueError):
    """Raised when a column cannot be cast to the expected dtype."""


def _enforce_field_dtype(df: pd.DataFrame, field: FieldSpec) -> pd.Series:
    """Return a Series cast to the field's expected dtype.

    No mutation occurs here; any casting failure is surfaced to callers.
    """

    if field.name not in df.columns:
        if field.required:
            raise DtypeEnforcementError(
                f"Missing required column '{field.name}' for dtype enforcement."
            )
        # Optional field absent: do not materialize it.
        raise KeyError(field.name)

    series = df[field.name]
    if field.dtype == "datetime64[ns]":
        # Enforce timezone-aware UTC timestamps in nanosecond precision.
        return pd.to_datetime(series, utc=True, errors="raise", unit=None)

    try:
        return series.astype(field.dtype, copy=True, errors="raise")
    except (TypeError, ValueError) as exc:  # pandas raises ValueError for invalid casts
        raise DtypeEnforcementError(
            f"Failed to cast column '{field.name}' to dtype '{field.dtype}': {exc}"
        ) from exc


def enforce_dtypes(df: pd.DataFrame, schema: FeedSchema, *, copy: bool = True) -> pd.DataFrame:
    """Enforce schema-defined dtypes on a DataFrame.

    Args:
        df: Input DataFrame loaded from parquet.
        schema: Feed schema defining expected column dtypes.
        copy: Whether to copy the DataFrame before casting. Defaults to True to
            avoid mutating caller-owned frames.

    Returns:
        A DataFrame with columns cast to their required dtypes. Optional columns
        not present in the input remain absent.

    Raises:
        DtypeEnforcementError: If required columns are missing or casts fail.
    """

    working_df = df.copy(deep=copy)
    coerced_columns: dict[str, pd.Series] = {}

    # First pass: attempt casting; collect coerced Series to avoid partial mutation.
    for field in schema.fields:
        try:
            coerced_columns[field.name] = _enforce_field_dtype(working_df, field)
        except KeyError:
            # Optional field absent; skip without materializing missing data.
            continue

    # Apply coerced columns to the working DataFrame.
    for name, series in coerced_columns.items():
        working_df[name] = series

    return working_df
