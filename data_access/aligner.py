"""Timestamp alignment utilities.

This module provides deterministic ordering and alignment of feeds on their
timestamp columns. It does not interpolate, resample, or fill missing data.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import pandas as pd

from .raw_schema import FeedSchema


class AlignmentError(ValueError):
    """Raised when timestamp alignment or ordering cannot be achieved."""


def sort_and_validate_timestamp(
    df: pd.DataFrame,
    schema: FeedSchema,
    *,
    remove_exact_duplicates: bool = False,
) -> pd.DataFrame:
    """Sort a DataFrame by timestamp and enforce monotonicity.

    Args:
        df: Input DataFrame to sort.
        schema: Feed schema containing the timestamp column name.
        remove_exact_duplicates: Whether to drop duplicate rows before
            validating timestamp monotonicity.

    Raises:
        AlignmentError: If timestamp column missing, duplicates persist when not
            allowed, or timestamps are non-monotonic.
    """

    ts_col = schema.timestamp_column
    if ts_col not in df.columns:
        raise AlignmentError(f"Timestamp column '{ts_col}' not found in frame.")

    working = df.sort_values(ts_col, kind="mergesort")  # stable sort

    if remove_exact_duplicates:
        working = working.drop_duplicates(ignore_index=True)

    ts_series = working[ts_col]
    duplicated_ts = ts_series.duplicated()
    if duplicated_ts.any():
        raise AlignmentError(
            f"Duplicate timestamps detected in '{schema.feed_name}' feed; "
            "enable duplicate removal explicitly if upstream allows it."
        )

    if not ts_series.is_monotonic_increasing:
        raise AlignmentError(
            f"Timestamps for feed '{schema.feed_name}' are not monotonic after sorting."
        )

    return working.reset_index(drop=True)


def align_feeds_on_timestamp(
    feeds: Mapping[str, Tuple[pd.DataFrame, FeedSchema]],
    *,
    remove_exact_duplicates: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Align multiple feeds by sorting them on their timestamp columns.

    Alignment here means each feed is independently sorted and validated, then
    indexed by the timestamp column for consistent downstream joins. No
    reindexing or interpolation is performed across feeds.

    Args:
        feeds: Mapping of feed name to a tuple of (DataFrame, FeedSchema).
        remove_exact_duplicates: Whether to drop exact duplicate rows prior to
            validation for all feeds.

    Returns:
        Dictionary mapping feed name to a timestamp-indexed DataFrame.

    Raises:
        AlignmentError: If any feed cannot satisfy timestamp ordering rules.
    """

    aligned: Dict[str, pd.DataFrame] = {}
    for feed_name, (df, schema) in feeds.items():
        if feed_name != schema.feed_name:
            raise AlignmentError(
                f"Feed name key '{feed_name}' does not match schema '{schema.feed_name}'."
            )

        sorted_df = sort_and_validate_timestamp(
            df,
            schema,
            remove_exact_duplicates=remove_exact_duplicates,
        )
        aligned[feed_name] = sorted_df.set_index(schema.timestamp_column, drop=False)

    return aligned
