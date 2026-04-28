"""Cleaner for Futures Volume Flow source datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd


class VolumeFlowCleanerError(Exception):
    """Raised when cleaned source data violates deterministic quality constraints."""


CRITICAL_COLS = ["meta__timestamp", "meta__sequence_id"]


def _normalize_timestamp_and_sequence(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for col in CRITICAL_COLS:
        if col not in df.columns:
            raise VolumeFlowCleanerError(f"{dataset_name}: missing {col}")

    cleaned = df.copy(deep=True)
    cleaned["meta__timestamp"] = pd.to_datetime(cleaned["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    cleaned["meta__sequence_id"] = pd.to_numeric(cleaned["meta__sequence_id"], errors="raise").astype("int64")

    return cleaned


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first")


def _sort_deterministically(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    ordered = df.sort_values(by=["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    if not ordered["meta__timestamp"].is_monotonic_increasing:
        raise VolumeFlowCleanerError(f"{dataset_name}: meta__timestamp is non-monotonic after cleaning")
    if not ordered["meta__sequence_id"].is_monotonic_increasing:
        raise VolumeFlowCleanerError(f"{dataset_name}: meta__sequence_id is non-monotonic after cleaning")

    return ordered


def _repair_non_critical_nulls(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    repaired = df.copy(deep=True)

    critical_nulls = {c: int(repaired[c].isna().sum()) for c in CRITICAL_COLS}
    if any(v > 0 for v in critical_nulls.values()):
        raise VolumeFlowCleanerError(f"{dataset_name}: critical nulls detected: {critical_nulls}")

    for col in repaired.columns:
        if col in CRITICAL_COLS:
            continue

        series = repaired[col]
        if pd.api.types.is_bool_dtype(series):
            repaired[col] = series.fillna(False)
        elif pd.api.types.is_numeric_dtype(series):
            repaired[col] = series.fillna(0)

    residual_nulls = {
        c: int(repaired[c].isna().sum())
        for c in repaired.columns
        if int(repaired[c].isna().sum()) > 0
    }
    if residual_nulls:
        raise VolumeFlowCleanerError(
            f"{dataset_name}: unresolved nulls remain after deterministic repair: {residual_nulls}"
        )

    return repaired


def _clean_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    cleaned = _normalize_timestamp_and_sequence(df, dataset_name)
    cleaned = _drop_exact_duplicates(cleaned)
    cleaned = _sort_deterministically(cleaned, dataset_name)
    cleaned = _repair_non_critical_nulls(cleaned, dataset_name)
    return cleaned


def clean_source_frames(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return cleaned copies for all volume-flow source datasets."""

    return {
        "trades_df": _clean_one(trades_df, "trades"),
        "orderflow_df": _clean_one(orderflow_df, "orderflow"),
    }
