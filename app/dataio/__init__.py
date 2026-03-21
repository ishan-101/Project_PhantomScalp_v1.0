# app/dataio/__init__.py
from __future__ import annotations
import datetime as dt
from typing import Optional, Dict, Any

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from .binance_spot import BinanceSpotClient
from .delta_options import DeltaOptionsClient

class DataBundle:
    """
    Convenience wrapper for backtests: fetches
      - Spot OHLCV (e.g., BTCUSDT 1m)
      - Option OHLCV or trades for a chosen instrument (ATM by your strategy, or pass explicit)
    """

    def __init__(
        self,
        spot_client: Optional[BinanceSpotClient] = None,
        options_client: Optional[DeltaOptionsClient] = None,
    ):
        self.spot = spot_client or BinanceSpotClient()
        self.opt = options_client or DeltaOptionsClient()

    def load_spot_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
    ):
        return self.spot.fetch_klines(symbol=symbol, interval=interval, start=start, end=end)

    def load_option_ohlcv(
        self,
        instrument: str,
        interval: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
    ):
        return self.opt.fetch_option_ohlc(instrument=instrument, interval=interval, start=start, end=end)

    def load_option_trades(
        self,
        instrument: str,
        start: dt.datetime | int,
        end: dt.datetime | int,
    ):
        return self.opt.fetch_option_trades(instrument=instrument, start=start, end=end)
