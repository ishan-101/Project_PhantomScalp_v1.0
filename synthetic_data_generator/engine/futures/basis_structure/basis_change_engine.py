"""Basis change engine."""

from __future__ import annotations

import pandas as pd


def add_basis_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    basis = pd.to_numeric(out["fut__perp_spot_basis"], errors="coerce").fillna(0.0)
    out["fut__basis_change"] = basis.diff().fillna(0.0).astype("float32")
    return out
