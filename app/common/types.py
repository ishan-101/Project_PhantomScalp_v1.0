from typing import Literal, TypedDict

Mode = Literal["backtest", "paper", "live"]

class Bar(TypedDict):
    ts: int  # epoch ms
    open: float
    high: float
    low: float
    close: float
    volume: float