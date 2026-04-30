"""Engine for fut__derivatives_stress_index."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DerivativesStressIndexEngineError(Exception):
    """Raised when stress-index computation cannot be completed safely."""


REQUIRED_INPUT_COLS = [
    "fut__funding_oi_stress",
    "fut__liquidation_pressure_index",
    "fut__leverage_pressure_index",
]


def _z_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=0))

    if std <= 1e-12 or not np.isfinite(std):
        return pd.Series(np.zeros(len(values), dtype="float32"), index=values.index)

    normalized = (values - mean) / std
    return normalized.fillna(0.0)


def add_derivatives_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise DerivativesStressIndexEngineError(f"Missing required input columns: {missing}")

    out = df.copy(deep=True)
    normalized_components = [_z_normalize(out[col]) for col in REQUIRED_INPUT_COLS]

    composite = pd.concat(normalized_components, axis=1).mean(axis=1)
    out["fut__derivatives_stress_index"] = (
        pd.to_numeric(composite, errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    )
    return out
