# app/ml/labels/regime.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict
from scipy import stats

def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index(pd.DatetimeIndex(df.index))
        except Exception:
            # fallback: create a DatetimeIndex from a 'datetime' column if present
            if 'datetime' in df.columns:
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['datetime'])))
            else:
                raise ValueError("Input df must have a DatetimeIndex or a 'datetime' column.")
    return df

def _align_features(df: pd.DataFrame, required: Dict[str, str]) -> pd.DataFrame:
    """Ensure required columns exist; if missing, add sensible fallbacks and print a short warning."""
    for col, reason in required.items():
        if col not in df.columns:
            print(f"[regime] warning: missing column '{col}' ({reason}). Filling fallback zeros/NaN.")
            if col.startswith('tc_') or col in ('tc_atr', 'tc_adx', 'tc_trend_strength'):
                df[col] = np.nan
            else:
                df[col] = 0
    return df

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rolling_majority(series: pd.Series, window: int, seed: int = 0) -> pd.Series:
    # rolling majority (mode); on ties choose random but deterministic based on seed
    rs = np.random.RandomState(seed)
    def mode_or_random(x):
        vals, counts = np.unique(x[~pd.isna(x)], return_counts=True)
        if len(vals) == 0:
            return np.nan
        maxc = counts.max()
        modes = vals[counts == maxc]
        if len(modes) == 1:
            return modes[0]
        # deterministic tie-break
        return modes[rs.randint(0, len(modes))]
    return series.rolling(window, min_periods=1).apply(lambda x: mode_or_random(x), raw=False)

def label_regime(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """
    Produce discrete market-regime labels:
      0 = flat/chop
      1 = bullish/trending-up
      2 = bearish/trending-down

    API:
      label_regime(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series

    Params allowed keys:
      atr_col="tc_atr", adx_col="tc_adx", adx_threshold=25,
      trend_strength_col="tc_trend_strength", trend_strength_threshold=0.5,
      vol_comp_col="tc_vol_comp_ratio", vote_window=5, seed=0, verbose=False
    """
    params = params or {}
    atr_col = params.get("atr_col", "tc_atr")
    adx_col = params.get("adx_col", "tc_adx")
    adx_threshold = params.get("adx_threshold", 25)
    trend_strength_col = params.get("trend_strength_col", "tc_trend_strength")
    trend_strength_threshold = params.get("trend_strength_threshold", 0.5)
    vol_comp_col = params.get("vol_comp_col", "tc_vol_comp_ratio")
    vote_window = params.get("vote_window", 5)
    seed = int(params.get("seed", 0))
    verbose = bool(params.get("verbose", False))

    # Defensive
    df = _ensure_indexed(df).copy()

    # ensure column names are strings to avoid numeric column name issues
    df.columns = df.columns.map(str)

    required = {
        atr_col: "average true range",
        adx_col: "ADX (trend strength)",
        trend_strength_col: "trend_strength indicator",
        vol_comp_col: "volume comparison ratio",
    }
    df = _align_features(df, required)

    # Basic safe close series
    if 'close' not in df.columns:
        print("[regime] warning: 'close' not in df; creating synthetic close from ohlc columns.")
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
            df['close'] = (df['open'] + df['high'] + df['low']) / 3.0
        else:
            df['close'] = np.nan

    # Normalize ATR safely
    tc_atr = pd.to_numeric(df.get(atr_col, pd.Series(np.nan, index=df.index)).ffill().bfill(), errors='coerce').astype(float)
    close = pd.to_numeric(df['close'].astype(float), errors='coerce')
    tc_atr_norm = tc_atr / (close.replace(0, np.nan)).abs()
    # Trend detection
    adx = pd.to_numeric(df.get(adx_col, pd.Series(np.nan, index=df.index)).ffill().bfill(), errors='coerce').astype(float)
    trend_strength = pd.to_numeric(df.get(trend_strength_col, pd.Series(np.nan, index=df.index)).ffill().bfill(), errors='coerce').astype(float)

    trending = (adx >= adx_threshold) & (trend_strength >= trend_strength_threshold)
    trending = trending.fillna(False)

    # Direction via EMAs - compute when possible, otherwise fallback to momentum of close
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    # safe tc_atr_norm quantile call
    try:
        atr_q75 = tc_atr_norm.quantile(0.75, interpolation='nearest')
    except Exception:
        atr_q75 = tc_atr_norm.median()
    direction_bull = (ema21 > ema50) & (tc_atr_norm < atr_q75)
    direction_bear = ~direction_bull

    # Build raw labels
    raw = pd.Series(0, index=df.index, dtype=int)
    raw[trending & direction_bull] = 1
    raw[trending & direction_bear] = 2

    # Smooth with rolling majority to avoid noise
    smoothed = _rolling_majority(raw, max(1, int(vote_window)), seed=seed)
    smoothed = smoothed.fillna(0).astype(int)
    smoothed.name = "regime_sig"

    # Verbose print
    if verbose:
        counts = smoothed.value_counts().to_dict()
        print(f"[regime] counts: {counts}")
        # show small sample indices of each class
        for cls in [0,1,2]:
            idxs = smoothed[smoothed==cls].index[:3].tolist()
            print(f"[regime] sample idxs for {cls}: {idxs}")

    # Unit checks
    assert isinstance(smoothed.index, pd.DatetimeIndex)
    assert len(smoothed) == len(df)

    return smoothed
