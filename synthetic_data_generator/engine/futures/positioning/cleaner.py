"""Cleaner for Futures Positioning source dependencies."""

from __future__ import annotations

from typing import Dict

import pandas as pd


class PositioningCleanerError(Exception):
    """Raised when dependency frames cannot be deterministically cleaned."""


CRITICAL_COLS = ["meta__timestamp", "meta__sequence_id"]


def _normalize_meta(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [c for c in CRITICAL_COLS if c not in df.columns]
    if missing:
        raise PositioningCleanerError(f"{dataset_name}: missing critical columns {missing}")

    cleaned = df.copy(deep=True)
    cleaned["meta__timestamp"] = pd.to_datetime(cleaned["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    try:
        cleaned["meta__sequence_id"] = cleaned["meta__sequence_id"].astype("int64")
    except Exception as exc:
        raise PositioningCleanerError(f"{dataset_name}: meta__sequence_id cannot coerce to int64") from exc

    return cleaned


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first")


def _sort(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    ordered = df.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)
    if not ordered["meta__timestamp"].is_monotonic_increasing:
        raise PositioningCleanerError(f"{dataset_name}: timestamp became non-monotonic")
    if not ordered["meta__sequence_id"].is_monotonic_increasing:
        raise PositioningCleanerError(f"{dataset_name}: sequence became non-monotonic")
    return ordered


def _repair_non_critical_nulls(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    repaired = df.copy(deep=True)

    critical_nulls = {c: int(repaired[c].isna().sum()) for c in CRITICAL_COLS}
    if any(v > 0 for v in critical_nulls.values()):
        raise PositioningCleanerError(f"{dataset_name}: critical nulls detected {critical_nulls}")

    for col in repaired.columns:
        if col in CRITICAL_COLS:
            continue
        if pd.api.types.is_bool_dtype(repaired[col]):
            repaired[col] = repaired[col].fillna(False)
        elif pd.api.types.is_numeric_dtype(repaired[col]):
            repaired[col] = repaired[col].fillna(0)

    residual = {c: int(repaired[c].isna().sum()) for c in repaired.columns if int(repaired[c].isna().sum()) > 0}
    if residual:
        raise PositioningCleanerError(f"{dataset_name}: unresolved nulls remain {residual}")

    return repaired


def _clean_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = _normalize_meta(df, dataset_name)
    out = _drop_exact_duplicates(out)
    out = _sort(out, dataset_name)
    out = _repair_non_critical_nulls(out, dataset_name)
    return out


def clean_source_frames(oi_df: pd.DataFrame, funding_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "oi_df": _clean_one(oi_df, "open_interest"),
        "funding_df": _clean_one(funding_df, "funding_rate"),
    }
