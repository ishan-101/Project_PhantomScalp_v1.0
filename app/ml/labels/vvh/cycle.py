# app/ml/labels/cycle.py
from typing import Optional
import pandas as pd
import numpy as np


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True)
        df = df.copy()
        df.index = idx
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.copy()
            df.index = df.index.tz_convert("UTC")
    return df


def _check_required_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def label_cycle(ohlcv_df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """
    Simple cycle-phase labeler using SMA(3) vs SMA(21) crossover.

    Outputs pandas.Series named "cycle" with integer dtype and index (UTC).
    Suggested encoding (documented here):
      - 0 : NEUTRAL / small difference between SMAs
      - 1 : RISING phase (SMA3 > SMA21)
      - 2 : FALLING phase (SMA3 < SMA21)

    The "tiny threshold" used to decide neutral region is computed relative to median close:
      threshold = tiny_factor * median(close)
    where tiny_factor defaults to 1e-3 (0.1% of median close). This keeps it scale-aware.
    """
    if params is None:
        params = {}
    df = ohlcv_df.copy()
    df = _ensure_datetime_index(df)
    _check_required_columns(df, ["open", "high", "low", "close", "volume"])

    close = df["close"].astype(float)

    sma_short = close.rolling(window=3, min_periods=1).mean()
    sma_long = close.rolling(window=21, min_periods=1).mean()

    tiny_factor = float(params.get("tiny_factor", 1e-3))
    threshold = tiny_factor * (close.median() if not np.isnan(close.median()) else 1.0)

    diff = sma_short - sma_long

    cond_neutral = diff.abs() < threshold
    cond_rising = (sma_short > sma_long) & (~cond_neutral)
    cond_falling = (sma_short < sma_long) & (~cond_neutral)

    cycle = pd.Series(0, index=df.index, name="cycle", dtype="int64")
    cycle.loc[cond_rising] = 1
    cycle.loc[cond_falling] = 2

    return cycle
