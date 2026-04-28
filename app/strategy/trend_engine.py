# app/strategy/trend_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrendState:
    direction: int = 0            # -1 down, 0 neutral, +1 up
    fresh: bool = False           # True when a new flip just happened
    strength: float = 0.0         # 0..1 composite strength
    last_flip_ts: Optional[int] = None
    reason: str = ""              # human-readable explanation


def _slope(a: float, b: float) -> float:
    # avoid div by zero; signed normalized slope proxy
    if a == 0 and b == 0:
        return 0.0
    return (b - a) / (abs(a) + abs(b) + 1e-9)


class TrendEngine:
    """
    Minimal deterministic core:
      - ribbon slope from fast/slow EMAs (9/21 and 50/200)
      - supertrend direction (bool: up/down)
      - recent structure break flag (BOS/CHOCH)
    Expects a dict 'ind' with precomputed indicators for the TF you run on.
    """

    def __init__(self,
                 ema_fast_key: str = "ema_9",
                 ema_slow_key: str = "ema_21",
                 ema_major1_key: str = "ema_50",
                 ema_major2_key: str = "ema_200",
                 supertrend_key: str = "supertrend_dir",
                 bos_key: str = "structure_bos",
                 strength_weights: Dict[str, float] = None):
        self.ema_fast_key = ema_fast_key
        self.ema_slow_key = ema_slow_key
        self.ema_major1_key = ema_major1_key
        self.ema_major2_key = ema_major2_key
        self.supertrend_key = supertrend_key
        self.bos_key = bos_key
        self.state = TrendState()
        self.weights = strength_weights or {"ribbon": 0.45, "major": 0.35, "bos": 0.20}

    def update(self, ts: int, price: float, ind: Dict[str, Any]) -> TrendState:
        # Required indicator fields present?
        for k in [self.ema_fast_key, self.ema_slow_key,
                  self.ema_major1_key, self.ema_major2_key,
                  self.supertrend_key, self.bos_key]:
            if k not in ind:
                raise KeyError(f"TrendEngine: missing indicator '{k}'")

        # Signals
        ribbon_up = ind[self.ema_fast_key] > ind[self.ema_slow_key]
        ribbon_down = ind[self.ema_fast_key] < ind[self.ema_slow_key]
        major_up = ind[self.ema_major1_key] > ind[self.ema_major2_key]
        major_down = ind[self.ema_major1_key] < ind[self.ema_major2_key]
        st_up = bool(ind[self.supertrend_key])  # True if up
        st_down = not st_up
        bos = int(bool(ind[self.bos_key]))      # 0/1 structure break with trend

        # Compute composite strength [0..1]
        ribbon_strength = 1.0 if ribbon_up else (1.0 if ribbon_down else 0.0)
        major_strength = 1.0 if major_up else (1.0 if major_down else 0.0)
        strength = (
            self.weights["ribbon"] * ribbon_strength
            + self.weights["major"] * major_strength
            + self.weights["bos"] * bos
        )
        strength = max(0.0, min(1.0, strength))

        # Determine direction (simple voting of signals)
        up_votes = int(ribbon_up) + int(major_up) + int(st_up)
        down_votes = int(ribbon_down) + int(major_down) + int(st_down)
        direction = 0
        if up_votes >= 2 and up_votes > down_votes:
            direction = +1
        elif down_votes >= 2 and down_votes > up_votes:
            direction = -1

        fresh = False
        reason = f"votes(up={up_votes},down={down_votes}) st={'UP' if st_up else 'DOWN'} bos={bos} strength={strength:.2f}"
        if direction != self.state.direction and direction != 0:
            fresh = True
            self.state.last_flip_ts = ts

        self.state.direction = direction
        self.state.fresh = fresh
        self.state.strength = strength
        self.state.reason = reason
        return self.state
