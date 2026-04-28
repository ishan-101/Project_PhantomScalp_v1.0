"""Cleaner for Futures Open Interest Engine source datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd


class OpenInterestCleanerError(Exception):
    """Raised when cleaned source data violates deterministic quality constraints."""


def _normalize_timestamp_and_sequence(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if "meta__timestamp" not in df.columns:
        raise OpenInterestCleanerError(f"{dataset_name}: missing meta__timestamp")
    if "meta__sequence_id" not in df.columns:
        raise OpenInterestCleanerError(f"{dataset_name}: missing meta__sequence_id")

    cleaned = df.copy(deep=True)
    cleaned["meta__timestamp"] = pd.to_datetime(cleaned["meta__timestamp"], utc=True)

    try:
        cleaned["meta__sequence_id"] = cleaned["meta__sequence_id"].astype("int64")
    except Exception as exc:
        raise OpenInterestCleanerError(
            f"{dataset_name}: meta__sequence_id cannot be coerced to int64"
        ) from exc

    return cleaned


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Canonical event identity across full row; deterministic keep='first'.
    return df.drop_duplicates(keep="first")


def _enforce_deterministic_order(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    ordered = df.sort_values(
        by=["meta__timestamp", "meta__sequence_id"],
        kind="mergesort",  # stable deterministic sorting
    ).reset_index(drop=True)

    if not ordered["meta__timestamp"].is_monotonic_increasing:
        raise OpenInterestCleanerError(
            f"{dataset_name}: meta__timestamp is non-monotonic after cleaning"
        )
    if not ordered["meta__sequence_id"].is_monotonic_increasing:
        raise OpenInterestCleanerError(
            f"{dataset_name}: meta__sequence_id is non-monotonic after cleaning"
        )

    return ordered


def _repair_non_critical_nulls(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    repaired = df.copy(deep=True)

    critical_cols = ["meta__timestamp", "meta__sequence_id"]
    critical_nulls = {c: int(repaired[c].isna().sum()) for c in critical_cols}
    if any(v > 0 for v in critical_nulls.values()):
        raise OpenInterestCleanerError(
            f"{dataset_name}: critical nulls detected: {critical_nulls}"
        )

    for col in repaired.columns:
        if col in critical_cols:
            continue

        series = repaired[col]
        if pd.api.types.is_bool_dtype(series):
            repaired[col] = series.fillna(False)
        elif pd.api.types.is_numeric_dtype(series):
            repaired[col] = series.fillna(0)

    # Disallow unresolved nulls in non-numeric/object-like helper columns unless explicitly approved.
    residual_nulls = {
        c: int(repaired[c].isna().sum())
        for c in repaired.columns
        if int(repaired[c].isna().sum()) > 0
    }
    if residual_nulls:
        raise OpenInterestCleanerError(
            f"{dataset_name}: unresolved nulls remain after deterministic repair: {residual_nulls}"
        )

    return repaired


def _clean_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    cleaned = _normalize_timestamp_and_sequence(df, dataset_name)
    cleaned = _drop_exact_duplicates(cleaned)
    cleaned = _enforce_deterministic_order(cleaned, dataset_name)
    cleaned = _repair_non_critical_nulls(cleaned, dataset_name)
    return cleaned


def clean_source_frames(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return cleaned copies for trades and orderflow source datasets."""

    cleaned_trades = _clean_one(trades_df, "trades")
    cleaned_orderflow = _clean_one(orderflow_df, "orderflow")

    return {
        "trades_df": cleaned_trades,
        "orderflow_df": cleaned_orderflow,
    }

