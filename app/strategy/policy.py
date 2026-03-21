# app/strategy/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

from app.strategy.trend_engine import TrendState
from app.ml.serve.signals import MlSignals


Profile = Literal["scalp", "runner", "trend"]

@dataclass
class PolicyConfig:
    regime_conf_min: float = 0.55
    reversal_p_max: float = 0.35
    impulse_boost_min: float = 0.60
    r_base_pct: float = 0.5  # % equity at risk baseline
    atr_stop_mult: Dict[Profile, float] = None
    trail_kama_mult: Dict[Profile, float] = None

    def __post_init__(self):
        self.atr_stop_mult = self.atr_stop_mult or {"scalp": 0.9, "runner": 1.2, "trend": 1.5}
        self.trail_kama_mult = self.trail_kama_mult or {"scalp": 1.1, "runner": 1.3, "trend": 1.6}


@dataclass
class Decision:
    action: Literal["BUY_CALL", "BUY_PUT", "FLAT"]
    size_risk_pct: float
    stop_atr_mult: float
    trail_kama_mult: float
    reason: str


class Policy:
    """
    Merge deterministic trend + ML signals into a decision for profiles:
    - scalp: low TF
    - runner: mid TF
    - trend: bias from high TF, entries on pullbacks (handled upstream)
    """

    def __init__(self, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()

    def decide(self,
               profile: Profile,
               trend: TrendState,
               ml: MlSignals,
               iv_percentile: Optional[float] = None,
               whipsaw_risk: float = 0.3) -> Decision:
        # Base rules
        if trend.direction == 0:
            return Decision("FLAT", 0.0, self.cfg.atr_stop_mult[profile],
                            self.cfg.trail_kama_mult[profile], "Neutral trend")

        # ML gates
        if ml.regime_conf < self.cfg.regime_conf_min:
            return Decision("FLAT", 0.0, self.cfg.atr_stop_mult[profile],
                            self.cfg.trail_kama_mult[profile],
                            f"Low regime confidence {ml.regime_conf:.2f}")

        if max(ml.reversal_p_3, ml.reversal_p_5) > self.cfg.reversal_p_max:
            return Decision("FLAT", 0.0, self.cfg.atr_stop_mult[profile],
                            self.cfg.trail_kama_mult[profile],
                            f"High reversal risk {max(ml.reversal_p_3, ml.reversal_p_5):.2f}")

        # Position sizing based on confidence, impulse, IV regime, and whipsaw
        size_mult = (0.5 + ml.regime_conf) * (0.5 + 0.8 * ml.impulse_p) * max(0.5, 1.2 - whipsaw_risk)
        if iv_percentile is not None:
            # De‑risk extremes of IV
            if iv_percentile < 5 or iv_percentile > 95:
                size_mult *= 0.8

        size_risk_pct = max(0.1, min(2.0, self.cfg.r_base_pct * size_mult))  # clamp to sane bounds

        # Action
        action = "BUY_CALL" if trend.direction > 0 else "BUY_PUT"
        reason = f"{profile} dir={trend.direction} fresh={trend.fresh} " \
                 f"conf={ml.regime_conf:.2f} imp={ml.impulse_p:.2f} size={size_risk_pct:.2f}%"
        return Decision(action, size_risk_pct,
                        self.cfg.atr_stop_mult[profile],
                        self.cfg.trail_kama_mult[profile],
                        reason)
