from dataclasses import dataclass

@dataclass
class Signal:
    ts: int | None
    direction: str  # long|short|flat
    strength: float

THRESH_LONG = 0.55
THRESH_SHORT = 0.45

def make_trade_signal(proba_long: float, row) -> Signal:
    if proba_long >= THRESH_LONG:
        return Signal(ts=None, direction="long", strength=proba_long)
    elif proba_long <= THRESH_SHORT:
        return Signal(ts=None, direction="short", strength=1 - proba_long)
    return Signal(ts=None, direction="flat", strength=0.0)