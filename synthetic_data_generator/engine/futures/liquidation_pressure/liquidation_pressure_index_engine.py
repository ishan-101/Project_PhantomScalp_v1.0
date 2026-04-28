"""Composite liquidation pressure index engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = [
    "fut__long_liquidation_volume",
    "fut__short_liquidation_volume",
    "fut__liquidation_imbalance",
    "fut__liquidation_cluster_distance",
    "fut__liquidation_velocity",
    "fut__funding_oi_stress",
    "fut__oi_velocity",
]


def add_liquidation_pressure_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Pressure index engine missing required columns: {missing}")

    long_liq = pd.to_numeric(out["fut__long_liquidation_volume"], errors="coerce").fillna(0.0)
    short_liq = pd.to_numeric(out["fut__short_liquidation_volume"], errors="coerce").fillna(0.0)
    imbalance = pd.to_numeric(out["fut__liquidation_imbalance"], errors="coerce").fillna(0.0)
    distance = pd.to_numeric(out["fut__liquidation_cluster_distance"], errors="coerce").fillna(0.0)
    velocity = pd.to_numeric(out["fut__liquidation_velocity"], errors="coerce").fillna(0.0)
    funding_stress = pd.to_numeric(out["fut__funding_oi_stress"], errors="coerce").fillna(0.0)
    oi_vel = pd.to_numeric(out["fut__oi_velocity"], errors="coerce").fillna(0.0)

    total_liq = long_liq + short_liq
    liq_intensity = np.tanh(total_liq / (total_liq.rolling(180, min_periods=5).median().replace(0.0, np.nan))).fillna(0.0)
    directional_stress = imbalance.abs().clip(0.0, 1.0)
    cluster_inverse = (1.0 / (1.0 + distance.clip(lower=0.0) * 120.0)).clip(0.0, 1.0)
    velocity_stress = np.tanh(velocity.abs())
    leverage_stress = np.tanh(funding_stress.clip(lower=0.0) + (-oi_vel).clip(lower=0.0) * 3.0)

    pressure = (
        0.30 * liq_intensity
        + 0.18 * directional_stress
        + 0.18 * cluster_inverse
        + 0.14 * velocity_stress
        + 0.20 * leverage_stress
    )

    out["fut__liquidation_pressure_index"] = (
        pd.Series(pressure, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0).astype("float32")
    )
    return out
