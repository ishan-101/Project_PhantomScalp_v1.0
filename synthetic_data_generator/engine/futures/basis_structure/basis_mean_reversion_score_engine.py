"""Basis mean-reversion score engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hours_to_next_funding(ts: pd.Series, interval_hours: int = 8) -> pd.Series:
    ns = pd.to_datetime(ts, utc=True).astype("int64")
    sec = (ns // 10**9).astype("int64")
    interval_sec = int(interval_hours * 3600)
    elapsed = sec % interval_sec
    remaining = (interval_sec - elapsed) % interval_sec
    return pd.Series(remaining / 3600.0, index=ts.index)


def add_basis_mean_reversion_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    basis = pd.to_numeric(out["fut__perp_spot_basis"], errors="coerce").fillna(0.0)
    z = pd.to_numeric(out["fut__basis_zscore"], errors="coerce").fillna(0.0)
    vel = pd.to_numeric(out["fut__basis_velocity"], errors="coerce").fillna(0.0)

    funding_rate = pd.to_numeric(out.get("fut__funding_rate", 0.0), errors="coerce").fillna(0.0)
    funding_pressure = pd.to_numeric(out.get("fut__funding_pressure_index", 0.0), errors="coerce").fillna(0.0)
    oi_stress = pd.to_numeric(out.get("fut__funding_oi_stress", 0.0), errors="coerce").fillna(0.0)
    alignment = pd.to_numeric(out.get("__basis_funding_pressure_alignment", 0.0), errors="coerce").fillna(0.0)

    hrs_to_funding = _hours_to_next_funding(out["meta__timestamp"], interval_hours=8)
    urgency = np.exp(-hrs_to_funding / 2.0)  # strongest in final ~2h before funding.

    extreme_component = np.tanh(z.abs() / 2.0)
    velocity_component = np.tanh((np.sign(basis) * vel.abs()) / 0.0015)

    # Opposing funding pressure supports mean reversion; aligned pressure supports continuation.
    funding_alignment = np.tanh(np.sign(basis) * (5.0 * funding_rate + 0.7 * funding_pressure + 0.4 * alignment))
    stress_amplifier = 0.6 + 0.4 * np.tanh(oi_stress.abs() * 2.0)

    mean_reversion_propensity = extreme_component * urgency * stress_amplifier
    continuation_propensity = np.tanh(0.65 * funding_alignment + 0.35 * velocity_component)

    # Sign convention: positive => premium compression expected, negative => discount deepening expected.
    score = np.sign(basis) * mean_reversion_propensity - continuation_propensity * (1.0 - urgency * 0.45)

    out["fut__basis_mean_reversion_score"] = (
        pd.Series(score, index=out.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0).astype("float32")
    )
    return out
