from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional

import pandas as pd


class NullHandlingStrategy(str, Enum):
    """Supported null-handling behaviors."""

    RAISE = "raise"
    ZERO_FILL = "zero_fill"
    FORWARD_FILL = "forward_fill"

    # 🔥 ADDED (schema-aligned)
    FORWARD_FILL_THEN_ZERO = "forward_fill_then_zero"
    COMPUTE_ELSE_ZERO = "compute_else_zero"
    COMPUTE_USING_BASELINE_STATS_ELSE_ZERO = "compute_using_baseline_stats_else_zero"
    COMPUTE_ELSE_ONE = "compute_else_1.0"


def _validate_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> list[str]:
    if columns is None:
        return list(df.columns)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not found for null handling: {missing}")
    return list(columns)


@dataclass
class NullDiagnostics:
    """Diagnostics about null handling operations."""

    initial_nulls: Mapping[str, int] = field(default_factory=dict)
    filled_nulls: Mapping[str, int] = field(default_factory=dict)
    final_nulls: Mapping[str, int] = field(default_factory=dict)


def apply_null_handling(
    df: pd.DataFrame,
    strategy: NullHandlingStrategy,
    columns: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, NullDiagnostics]:

    target_cols = _validate_columns(df, columns)
    working_df = df.copy()

    initial_nulls = {col: int(working_df[col].isna().sum()) for col in target_cols}

    # ===== EXISTING LOGIC (UNCHANGED) =====

    if strategy == NullHandlingStrategy.RAISE:
        remaining = {col: count for col, count in initial_nulls.items() if count > 0}
        if remaining:
            raise ValueError(
                "Nulls detected with 'raise' strategy: "
                + ", ".join(f"{col}={count}" for col, count in remaining.items())
            )
        diagnostics = NullDiagnostics(initial_nulls=initial_nulls, filled_nulls={}, final_nulls=initial_nulls)
        return working_df, diagnostics

    if strategy == NullHandlingStrategy.ZERO_FILL:
        for col in target_cols:
            if initial_nulls[col] > 0:
                working_df[col] = working_df[col].fillna(0)

    elif strategy == NullHandlingStrategy.FORWARD_FILL:
        for col in target_cols:
            if initial_nulls[col] > 0:
                working_df[col] = working_df[col].ffill()

    # ===== 🔥 NEW LOGIC (ADDED ONLY) =====

    elif strategy == NullHandlingStrategy.FORWARD_FILL_THEN_ZERO:
        for col in target_cols:
            working_df[col] = working_df[col].ffill().fillna(0)

    elif strategy == NullHandlingStrategy.COMPUTE_ELSE_ZERO:
        for col in target_cols:
            working_df[col] = working_df[col].fillna(0)

    elif strategy == NullHandlingStrategy.COMPUTE_USING_BASELINE_STATS_ELSE_ZERO:
        for col in target_cols:
            working_df[col] = working_df[col].fillna(0)

    elif strategy == NullHandlingStrategy.COMPUTE_ELSE_ONE:
        for col in target_cols:
            working_df[col] = working_df[col].fillna(1.0)

    else:
        raise ValueError(f"Unsupported null handling strategy: {strategy}")

    # ===== EXISTING LOGIC CONTINUES =====

    filled_nulls = {
        col: initial_nulls[col] - int(working_df[col].isna().sum())
        for col in target_cols
    }
    final_nulls = {col: int(working_df[col].isna().sum()) for col in target_cols}

    diagnostics = NullDiagnostics(
        initial_nulls=initial_nulls,
        filled_nulls=filled_nulls,
        final_nulls=final_nulls,
    )

    return working_df, diagnostics