# app/ml/labels/reversal.py
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


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def label_reversal(ohlcv_df: pd.DataFrame, features_df: Optional[pd.DataFrame] = None, params: Optional[dict] = None) -> pd.Series:
    """
    Label potential reversals using price exhaustion + optional microstructure signals.

    Returns pandas.Series indexed by the (UTC) datetime index with integer dtype.
    Encoding: {0: no-reversal, 1: reversal-signal}
    Series name: "reversal"

    Algorithm (heuristic):
      - Price exhaustion: detect short-window local highs/lows (3-10 bars) combined with RSI divergence (RSI falling from high for tops, rising from low for bottoms).
      - If features_df provided, require price exhaustion AND at least one microstructure trigger:
          * ms_l3_momentum_burst abs >= 1.0
          * abs(ms_aggression_idx_v2) > 0.5
        In addition, a conservative fallback: if a strong microstructure trigger exists AND price is very near a short-window high/low (within a tiny tolerance), label reversal (this addresses data where RSI divergence is weak but microstructure clearly signals exhaustion).
      - If features_df is None, require stronger price-only signals (to avoid false positives).
    """
    if params is None:
        params = {}
    df = ohlcv_df.copy()
    df = _ensure_datetime_index(df)
    _check_required_columns(df, ["open", "high", "low", "close", "volume"])

    close = df["close"].astype(float)
    rsi = _rsi(close, period=14)

    # Price exhaustion windows
    short_w = int(params.get("short_window", 5))  # 3-10 typical; default 5
    strong_w = int(params.get("strong_window", 10))
    price_near_tol = float(params.get("price_near_tol", 0.01))  # 1% default tolerance for "near extreme"

    # Identify local highs and lows using rolling max/min (vectorized)
    rolling_max = close.rolling(window=short_w, min_periods=1).max()
    rolling_min = close.rolling(window=short_w, min_periods=1).min()

    # Candidate top exhaustion: close equals rolling_max and RSI below some threshold or falling
    is_local_top = close >= rolling_max
    rsi_falling = rsi.diff() < 0
    top_price_condition = is_local_top & (rsi < 70) & rsi_falling

    # Candidate bottom exhaustion
    is_local_bottom = close <= rolling_min
    rsi_rising = rsi.diff() > 0
    bottom_price_condition = is_local_bottom & (rsi > 30) & rsi_rising

    price_signal = (top_price_condition | bottom_price_condition)

    # If features_df provided, check microstructure triggers
    micro_trigger = pd.Series(False, index=df.index)
    if features_df is not None:
        f = features_df.copy()
        # ensure index alignment and datetime index in UTC
        if "datetime" in f.columns:
            f.index = pd.to_datetime(f["datetime"], utc=True)
        if not isinstance(f.index, pd.DatetimeIndex):
            raise ValueError("features_df must have a DatetimeIndex or 'datetime' column.")
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        else:
            f.index = f.index.tz_convert("UTC")

        # align to price index
        f = f.reindex(df.index, method=None).fillna(0.0)

        # microstructure heuristics
        m_burst = f.get("ms_l3_momentum_burst", pd.Series(0.0, index=f.index)).astype(float)
        m_aggr = f.get("ms_aggression_idx_v2", pd.Series(0.0, index=f.index)).astype(float)

        # strong micro triggers: big signed burst OR strong aggression index
        micro_trigger = (m_burst.abs() >= 1.0) | (m_aggr.abs() > 0.5)

        # Additional conservative fallback: if micro_trigger AND price is very near the short-window high/low,
        # allow reversal even if RSI divergence wasn't strong.
        # compute relative proximity to rolling extreme
        # avoid divide-by-zero by comparing absolute diffs relative to rolling_max/min
        near_top = ((rolling_max - close).abs() / rolling_max.replace(0, np.nan)) <= price_near_tol
        near_bottom = ((close - rolling_min).abs() / rolling_min.replace(0, np.nan)) <= price_near_tol
        price_near_extreme = (near_top | near_bottom).fillna(False)

        # Final micro-empowered signal: either (price_signal AND micro_trigger) OR (micro_trigger AND price_near_extreme)
        final_micro_signal = (price_signal & micro_trigger) | (micro_trigger & price_near_extreme)

    # Final rule:
    # - If features_df provided: prefer final_micro_signal (conservative)
    # - If features_df not provided: require stronger price-only signal (use longer lookback and more extreme RSI)
    if features_df is not None:
        final_signal = final_micro_signal.fillna(False)
    else:
        # stronger price-only conditions
        rolling_max_strong = close.rolling(window=strong_w, min_periods=1).max()
        rolling_min_strong = close.rolling(window=strong_w, min_periods=1).min()
        is_top_strong = close >= rolling_max_strong
        is_bottom_strong = close <= rolling_min_strong
        rsi_strong_top = (rsi < 65) & (rsi.diff() < 0)
        rsi_strong_bot = (rsi > 35) & (rsi.diff() > 0)
        final_signal = (is_top_strong & rsi_strong_top) | (is_bottom_strong & rsi_strong_bot)

    result = pd.Series(0, index=df.index, name="reversal", dtype="int64")
    result.loc[final_signal.fillna(False)] = 1
    return result
