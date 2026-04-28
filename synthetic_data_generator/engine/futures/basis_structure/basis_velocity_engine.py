"""Basis velocity engine."""

from __future__ import annotations

import pandas as pd


def add_basis_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    change = pd.to_numeric(out["fut__basis_change"], errors="coerce").fillna(0.0)
    seq = pd.to_numeric(out["meta__sequence_id"], errors="coerce").fillna(0.0)
    step = seq.diff().abs().replace(0.0, 1.0).fillna(1.0)
    out["fut__basis_velocity"] = (change / step).replace([float("inf"), float("-inf")], 0.0).fillna(0.0).astype("float32")
    return out
