# app/ml/serve/signals.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MlSignals:
    """
    ML outputs consumed by Policy. All fields are floats in [0,1] unless noted.
    - regime: -1 (down), 0 (range), +1 (up)
    - regime_conf: model confidence in regime classification
    - reversal_p_*: probability of reversal within N bars
    - cycle_phase: -1..+1 latent phase (optional usage)
    - impulse_p: probability of an 'impulse window' in next T minutes
    """
    regime: int                  # -1, 0, +1
    regime_conf: float           # 0..1
    reversal_p_3: float          # 0..1
    reversal_p_5: float          # 0..1
    cycle_phase: float           # -1..+1
    impulse_p: float             # 0..1

    @staticmethod
    def conservative_default() -> "MlSignals":
        # Safe defaults before models are wired
        return MlSignals(regime=0, regime_conf=0.5, reversal_p_3=0.5,
                         reversal_p_5=0.5, cycle_phase=0.0, impulse_p=0.5)
