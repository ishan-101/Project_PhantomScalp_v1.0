"""Cleaner for Futures Basis Structure source datasets."""

from __future__ import annotations

from typing import Dict

import pandas as pd


class BasisStructureCleanerError(Exception):
    """Raised when cleaned source data violates deterministic constraints."""


CRITICAL_COLS = ["meta__timestamp", "meta__sequence_id"]


def _clean_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for c in CRITICAL_COLS:
        if c not in df.columns:
            raise BasisStructureCleanerError(f"{dataset_name}: missing {c}")

    out = df.copy(deep=True)
    out["meta__timestamp"] = pd.to_datetime(out["meta__timestamp"], utc=True).astype("datetime64[ns, UTC]")
    out["meta__sequence_id"] = pd.to_numeric(out["meta__sequence_id"], errors="coerce")

    if out[CRITICAL_COLS].isna().any().any():
        bad = {c: int(out[c].isna().sum()) for c in CRITICAL_COLS if int(out[c].isna().sum()) > 0}
        raise BasisStructureCleanerError(f"{dataset_name}: critical nulls detected: {bad}")

    out["meta__sequence_id"] = out["meta__sequence_id"].astype("int64")
    out = out.drop_duplicates(keep="first")
    out = out.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    for col in out.columns:
        if col in CRITICAL_COLS:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].fillna(False)

    residual = {c: int(out[c].isna().sum()) for c in out.columns if int(out[c].isna().sum()) > 0}
    if residual:
        raise BasisStructureCleanerError(f"{dataset_name}: unresolved nulls remain after repair: {residual}")

    return out


def clean_source_frames(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, funding_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "trades_df": _clean_one(trades_df, "trades"),
        "orderflow_df": _clean_one(orderflow_df, "orderflow"),
        "funding_df": _clean_one(funding_df, "funding_rate"),
    }
