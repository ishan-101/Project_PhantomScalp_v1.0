# feature_engineering/utils/feature_helpers.py

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd

from .null_handling import apply_null_handling, NullHandlingStrategy
from .dtype_enforcement import enforce_dtypes


# ---------------------------------------------------------
# NULL POLICY ADAPTER (schema → existing strategies)
# ---------------------------------------------------------
def apply_schema_null_policy(
    df: pd.DataFrame,
    columns: Iterable[str],
    null_policy: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Adapts schema-level null_policy strings to existing NullHandlingStrategy.
    We DO NOT modify null_handling.py; we map here.

    Supported (based on your schema):
      - "forward_fill_then_zero"
      - "compute_else_zero"
      - "compute_else_one"
      - "compute_using_baseline_stats_else_zero"
      - "zero_if_missing"
      - "false_if_missing"
      - "compute_else_forward_fill"
      - "raise"
    """

    working = df.copy()

    # pre-clean infinities for all compute_* policies
    if null_policy.startswith("compute_"):
        for c in columns:
            working[c] = working[c].replace([np.inf, -np.inf], np.nan)

    if null_policy == "forward_fill_then_zero":
        working, d1 = apply_null_handling(working, NullHandlingStrategy.FORWARD_FILL, columns)
        working, d2 = apply_null_handling(working, NullHandlingStrategy.ZERO_FILL, columns)
        return working, {"ffill": d1, "zero": d2}

    if null_policy in ("compute_else_zero", "compute_using_baseline_stats_else_zero", "zero_if_missing"):
        return apply_null_handling(working, NullHandlingStrategy.ZERO_FILL, columns)

    if null_policy == "compute_else_one":
        for c in columns:
            working[c] = working[c].fillna(1.0)
        return working, {"filled_with_one": True}

    if null_policy == "false_if_missing":
        for c in columns:
            working[c] = working[c].fillna(False)
        return working, {"filled_with_false": True}

    if null_policy == "compute_else_forward_fill":
        return apply_null_handling(working, NullHandlingStrategy.FORWARD_FILL, columns)

    if null_policy == "raise":
        return apply_null_handling(working, NullHandlingStrategy.RAISE, columns)

    raise ValueError(f"Unsupported null_policy: {null_policy}")


# ---------------------------------------------------------
# COMMON CLEAN STEP
# ---------------------------------------------------------
def clean_compute_output(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Replace inf/-inf with NaN before null policy.
    """
    working = df.copy()
    for c in columns:
        working[c] = working[c].replace([np.inf, -np.inf], np.nan)
    return working


# ---------------------------------------------------------
# DTYPE + ZERO-NULL ENFORCEMENT
# ---------------------------------------------------------
def finalize_frame(
    df: pd.DataFrame,
    expected_dtypes: Dict[str, str],
) -> pd.DataFrame:
    """
    Enforce dtypes strictly and assert zero-null guarantee.
    """
    out = enforce_dtypes(df, expected_dtypes)

    # zero-null guarantee
    nulls = int(out.isna().sum().sum())
    if nulls != 0:
        raise ValueError(f"Zero-null guarantee violated. Remaining nulls: {nulls}")

    return out