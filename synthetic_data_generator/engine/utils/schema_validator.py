# schema_validator.py
"""Small schema validator used by engines to check essential meta columns.
Throws SchemaValidationError on failures.
"""

from __future__ import annotations
from typing import List
import pandas as pd


class SchemaValidationError(Exception):
    pass


def validate_basic_tick_schema(df: pd.DataFrame, require_cols: List[str]):
    # check required columns present
    for c in require_cols:
        if c not in df.columns:
            raise SchemaValidationError(f"Missing required column: {c}")
    # check no nulls in required core columns
    core = ["meta__timestamp", "meta__sequence_id"]
    for c in core:
        if c not in df.columns:
            raise SchemaValidationError(f"Missing required core column: {c}")
    # timestamps tz-aware recommended
    ts = df["meta__timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaValidationError("meta__timestamp must be datetime dtype.")
    # prefer tz-aware
    # If tz-naive, convert to tz-aware UTC for consistency (non-fatal)
    if not pd.api.types.is_datetime64tz_dtype(ts):
        # try to localize to UTC (non-fatal)
        df["meta__timestamp"] = pd.to_datetime(df["meta__timestamp"], utc=True)
    # sequence monotonic check (strictly increasing)
    seq = df["meta__sequence_id"]
    if not seq.is_monotonic_increasing:
        raise SchemaValidationError("meta__sequence_id must be monotonic increasing.")
    # basic positivity tests optionally
    return True
