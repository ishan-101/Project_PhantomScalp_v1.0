"""Reusable validation helpers for feature engineering infrastructure."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when validation checks fail."""


def check_value_range(
    series: pd.Series,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True,
) -> None:
    """Validate that series values lie within the specified bounds.

    Args:
        series: Series to validate.
        min_value: Inclusive or exclusive lower bound.
        max_value: Inclusive or exclusive upper bound.
        inclusive: Whether bounds are inclusive.

    Raises:
        ValidationError: If values fall outside the specified range.
    """
    if min_value is not None:
        if inclusive:
            invalid = series < min_value
        else:
            invalid = series <= min_value
        if invalid.any():
            raise ValidationError(
                f"Values below minimum bound ({min_value}) detected in '{series.name}'."
            )

    if max_value is not None:
        if inclusive:
            invalid = series > max_value
        else:
            invalid = series >= max_value
        if invalid.any():
            raise ValidationError(
                f"Values above maximum bound ({max_value}) detected in '{series.name}'."
            )


def check_monotonic(
    series: pd.Series,
    *,
    increasing: bool = True,
    strict: bool = False,
) -> None:
    """Assert monotonicity of a series.

    Args:
        series: Series to validate for monotonic ordering.
        increasing: Expect increasing (True) or decreasing (False) order.
        strict: Require strictly monotonic progression if True.

    Raises:
        ValidationError: If monotonicity expectations are violated.
    """
    values = series.to_numpy()
    diffs = np.diff(values)
    if increasing:
        violation = diffs <= 0 if strict else diffs < 0
    else:
        violation = diffs >= 0 if strict else diffs > 0

    if violation.any():
        direction = "increasing" if increasing else "decreasing"
        strictness = "strictly " if strict else ""
        raise ValidationError(
            f"Series '{series.name}' is not {strictness}{direction} monotonic."
        )


def check_shape(
    data: pd.DataFrame | np.ndarray | Sequence[Sequence[object]],
    *,
    expected_rows: Optional[int] = None,
    expected_columns: Optional[int] = None,
) -> None:
    """Validate the shape of tabular data.

    Args:
        data: DataFrame, ndarray, or 2D sequence to inspect.
        expected_rows: If provided, require this exact row count.
        expected_columns: If provided, require this exact column count.

    Raises:
        ValidationError: If shapes do not match expectations.
    """
    if isinstance(data, pd.DataFrame):
        rows, cols = data.shape
    else:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValidationError("Data must be 2-dimensional for shape validation.")
        rows, cols = arr.shape

    if expected_rows is not None and rows != expected_rows:
        raise ValidationError(f"Expected {expected_rows} rows, found {rows}.")
    if expected_columns is not None and cols != expected_columns:
        raise ValidationError(f"Expected {expected_columns} columns, found {cols}.")


def check_duplicates(
    df: pd.DataFrame, *, subset: Optional[Iterable[str]] = None
) -> pd.Index:
    """Detect duplicate rows in a DataFrame.

    Args:
        df: DataFrame to inspect.
        subset: Optional columns to consider for duplication.

    Returns:
        Index of duplicate row positions.

    Raises:
        ValidationError: If duplicates are present.
    """
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicates = df.index[duplicate_mask]
    if not duplicates.empty:
        cols = list(subset) if subset is not None else list(df.columns)
        raise ValidationError(
            f"Duplicate rows detected when considering columns {cols}: indices {list(duplicates)}"
        )
    return duplicates
