"""Cleaner for Futures Leverage Metrics source datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd


class LeverageMetricsCleanerError(Exception):
    """Raised when cleaned leverage sources violate deterministic quality constraints."""


CRITICAL_COLS = ["meta__timestamp", "meta__sequence_id"]


def _clean_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for col in CRITICAL_COLS:
        if col not in df.columns:
            raise LeverageMetricsCleanerError(f"{dataset_name}: missing {col}")

    cleaned = df.copy(deep=True)
    cleaned["meta__timestamp"] = pd.to_datetime(cleaned["meta__timestamp"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    cleaned["meta__sequence_id"] = pd.to_numeric(cleaned["meta__sequence_id"], errors="coerce")

    critical_nulls = {c: int(cleaned[c].isna().sum()) for c in CRITICAL_COLS}
    if any(v > 0 for v in critical_nulls.values()):
        raise LeverageMetricsCleanerError(f"{dataset_name}: critical nulls detected: {critical_nulls}")

    cleaned["meta__sequence_id"] = cleaned["meta__sequence_id"].astype("int64")
    cleaned = cleaned.drop_duplicates(keep="first")
    cleaned = cleaned.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    if not cleaned["meta__timestamp"].is_monotonic_increasing:
        raise LeverageMetricsCleanerError(f"{dataset_name}: meta__timestamp non-monotonic after cleaning")
    if not cleaned["meta__sequence_id"].is_monotonic_increasing:
        raise LeverageMetricsCleanerError(f"{dataset_name}: meta__sequence_id non-monotonic after cleaning")

    residual = {c: int(cleaned[c].isna().sum()) for c in cleaned.columns if int(cleaned[c].isna().sum()) > 0}
    if residual:
        raise LeverageMetricsCleanerError(f"{dataset_name}: unresolved nulls remain: {residual}")

    return cleaned


def clean_source_frames(
    oi_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    basis_df: pd.DataFrame,
    volume_flow_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    return {
        "oi_df": _clean_one(oi_df, "open_interest"),
        "funding_df": _clean_one(funding_df, "funding_rate"),
        "basis_df": _clean_one(basis_df, "basis_structure"),
        "volume_flow_df": _clean_one(volume_flow_df, "volume_flow"),
    }
