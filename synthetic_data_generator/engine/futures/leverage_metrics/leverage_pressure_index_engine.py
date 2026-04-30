"""Engine for fut__leverage_pressure_index."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LeveragePressureIndexError(Exception):
    """Raised when required interaction features are missing."""


def _to_unit_variance(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
    std = float(values.std(ddof=0))
    if std <= eps:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=series.index, dtype="float32")
    normalized = (values / std).astype("float32")
    return normalized


def add_leverage_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "fut__oi_volume_interaction",
        "fut__funding_oi_interaction",
        "fut__basis_oi_interaction",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise LeveragePressureIndexError(f"Missing required columns for fut__leverage_pressure_index: {missing}")

    out = df.copy(deep=True)
    normalized_components = [
        _to_unit_variance(out["fut__oi_volume_interaction"]),
        _to_unit_variance(out["fut__funding_oi_interaction"]),
        _to_unit_variance(out["fut__basis_oi_interaction"]),
    ]
    comp_df = pd.concat(normalized_components, axis=1)
    out["fut__leverage_pressure_index"] = comp_df.mean(axis=1).astype("float32")
    out["fut__leverage_pressure_index"] = (
        out["fut__leverage_pressure_index"]
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .astype("float32")
    )
    return out
