"""Basis regime flag engine."""

from __future__ import annotations

import pandas as pd


def add_basis_regime_flag(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    out = df.copy(deep=True)
    z = pd.to_numeric(out["fut__basis_zscore"], errors="coerce").fillna(0.0)
    out["fut__basis_regime_flag"] = (z.abs() >= float(z_threshold)).astype(bool)
    return out
