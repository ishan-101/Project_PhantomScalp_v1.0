"""Deterministic near-term liquidation cascade probability estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = [
    "fut__liquidation_cluster_distance",
    "fut__funding_oi_stress",
    "fut__oi_velocity",
    "__liq__orderflow_aggression",
    "__liq__toxicity_proxy",
]


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def add_liquidation_cascade_probability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Cascade probability engine missing required columns: {missing}")

    distance = pd.to_numeric(out["fut__liquidation_cluster_distance"], errors="coerce").fillna(0.0)
    funding_stress = pd.to_numeric(out["fut__funding_oi_stress"], errors="coerce").fillna(0.0)
    oi_velocity = pd.to_numeric(out["fut__oi_velocity"], errors="coerce").fillna(0.0)
    aggression = pd.to_numeric(out["__liq__orderflow_aggression"], errors="coerce").fillna(0.0)
    toxicity = pd.to_numeric(out["__liq__toxicity_proxy"], errors="coerce").fillna(0.0)

    cluster_risk = 1.0 / (1.0 + 140.0 * distance.clip(lower=0.0))
    leverage_risk = np.tanh(funding_stress.clip(lower=0.0) + (-oi_velocity).clip(lower=0.0) * 3.5)
    aggression_risk = np.tanh(aggression.abs())
    toxicity_risk = np.tanh(toxicity.clip(lower=0.0))

    linear_score = (
        2.6 * cluster_risk
        + 1.6 * leverage_risk
        + 1.1 * aggression_risk
        + 0.9 * toxicity_risk
        - 2.7
    )

    probability = _sigmoid(pd.Series(linear_score, index=out.index))
    out["fut__liquidation_cascade_probability"] = probability.clip(0.0, 1.0).astype("float32")
    return out
