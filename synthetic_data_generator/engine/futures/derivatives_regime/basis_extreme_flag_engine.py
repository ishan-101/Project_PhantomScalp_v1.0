"""Engine for fut__basis_extreme_flag."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BasisExtremeFlagEngineError(Exception):
    """Raised when basis-extreme regime flag cannot be computed."""


REQUIRED_INPUT_COLS = ["fut__basis_zscore", "fut__basis_regime_flag"]
INSTITUTIONAL_ZSCORE_THRESHOLD = 2.5


def add_basis_extreme_flag(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise BasisExtremeFlagEngineError(f"Missing required input columns: {missing}")

    out = df.copy(deep=True)
    basis_z = pd.to_numeric(out["fut__basis_zscore"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regime = out["fut__basis_regime_flag"].astype("bool")

    threshold_hit = basis_z.abs() > INSTITUTIONAL_ZSCORE_THRESHOLD
    out["fut__basis_extreme_flag"] = (threshold_hit & regime).astype("bool")
    return out
