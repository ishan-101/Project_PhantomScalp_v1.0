"""Basis compression ratio engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_basis_compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    ts = pd.to_datetime(out["meta__timestamp"], utc=True)
    basis_abs = pd.to_numeric(out["fut__perp_spot_basis"], errors="coerce").fillna(0.0).abs()

    ordered = pd.DataFrame({"ts": ts, "abs_basis": basis_abs, "idx": out.index})
    ordered = ordered.sort_values("ts", kind="mergesort")
    roll = (
        ordered.set_index("ts")["abs_basis"].rolling("7D", min_periods=64).mean().shift(1).replace(0.0, np.nan)
    )
    ordered["ratio"] = (ordered["abs_basis"] / roll.values).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    ratio_series = ordered.set_index("idx")["ratio"].reindex(out.index)
    out["fut__basis_compression_ratio"] = pd.to_numeric(ratio_series, errors="coerce").fillna(1.0).clip(0.0, 10.0).astype("float32")
    return out
