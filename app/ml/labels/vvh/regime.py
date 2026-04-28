# app/ml/labels/regime.py
from typing import Optional
import pandas as pd
import numpy as np


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a UTC DatetimeIndex. If 'datetime' column exists it will be used.
    If index is DatetimeIndex but tz-naive it will be localized to UTC.
    """
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True)
        df = df.copy()
        df.index = idx
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column.")
        # make tz-aware in UTC
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


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute a simple ADX (14) using classic +DM/-DM/TR and rolling sums.
    If not enough data, result will contain NaNs; caller should handle fallback.
    This is vectorized and deterministic.
    Returns ADX as plain numeric series (not scaled to 0-100).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    high_diff = high.diff()
    low_diff = low.diff()

    plus_dm = np.where((high_diff > 0) & (high_diff > abs(low_diff)), high_diff, 0.0)
    minus_dm = np.where((low_diff > 0) & (low_diff > abs(high_diff)), low_diff, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing using rolling sum as approximation (vectorized)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus = pd.Series(plus_dm, index=df.index).rolling(window=period, min_periods=period).mean()
    minus = pd.Series(minus_dm, index=df.index).rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    denom = plus + minus
    denom = denom.replace(0, np.nan)

    plus_di = 100.0 * (plus / atr)
    minus_di = 100.0 * (minus / atr)

    dx = 100.0 * ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) )
    # ADX -- take rolling mean of DX as a simple smoothed ADX
    adx = dx.rolling(window=period, min_periods=period).mean()
    # ADX may be NaN for leading rows; return as-is
    return adx.fillna(np.nan)


def label_regime(ohlcv_df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """
    Label market regime using EMA cross and ADX strength.

    Returns pandas.Series indexed by the (UTC) datetime index with integer dtype.
    Encoding: {0: FLAT, 1: BULL/TREND_UP, 2: BEAR/TREND_DOWN}
    Series name: "regime"

    Heuristic:
      - EMA20 > EMA50 and ADX >= 20 -> 1 (BULL)
      - EMA20 < EMA50 and ADX >= 20 -> 2 (BEAR)
      - else -> 0 (FLAT)

    Defensive checks: requires columns ['open','high','low','close','volume'].
    """
    if params is None:
        params = {}
    df = ohlcv_df.copy()
    df = _ensure_datetime_index(df)
    _check_required_columns(df, ["open", "high", "low", "close", "volume"])

    close = df["close"].astype(float)

    ema20 = _ema(close, span=20)
    ema50 = _ema(close, span=50)

    # ADX
    adx = _compute_adx(df, period=14)

    # Fallback if ADX is mostly NaN: use absolute EMA20 slope magnitude as proxy
    adx_fallback = (ema20.diff().abs()).rolling(window=14, min_periods=1).mean()
    # scale fallback to roughly ADX-like scale — normalize
    # avoid division by zero
    median_fallback = max(1e-9, adx_fallback.median() if not np.isnan(adx_fallback.median()) else 1.0)
    adx_fallback_scaled = (adx_fallback / median_fallback) * 20.0  # map median to ~20

    adx_use = adx.copy()
    # where adx is NaN (insufficient history), use fallback
    adx_use = adx_use.fillna(adx_fallback_scaled)

    # Build regime labels
    cond_bull = (ema20 > ema50) & (adx_use >= 20)
    cond_bear = (ema20 < ema50) & (adx_use >= 20)

    regime = pd.Series(0, index=df.index, name="regime", dtype="int64")
    regime.loc[cond_bull] = 1
    regime.loc[cond_bear] = 2

    return regime
