# app/ml/labels/regime.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with a DatetimeIndex. If impossible, create a sane UTC minute index."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    tmp = df.copy()
    if "datetime" in tmp.columns:
        try:
            tmp["datetime"] = pd.to_datetime(tmp["datetime"])
            tmp = tmp.set_index("datetime")
            return tmp
        except Exception:
            pass
    # try coerce index to datetime
    try:
        tmp.index = pd.to_datetime(tmp.index)
        return tmp
    except Exception:
        # fallback to minute-grid ending now (UTC)
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(tmp), freq="min")
        tmp.index = idx
        return tmp


def _align_features(df: pd.DataFrame, required: Dict[str, str]) -> pd.DataFrame:
    """Ensure required columns exist; if missing, add sensible fallbacks and print a short warning."""
    for col, reason in required.items():
        if col not in df.columns:
            print(f"[regime] warning: missing column '{col}' ({reason}). Filling fallback zeros/NaN.")
            # fallback choice: indicators -> NaN, others -> 0
            if isinstance(col, str) and col.startswith("tc_"):
                df[col] = np.nan
            else:
                df[col] = 0
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rolling_majority(series: pd.Series, window: int, seed: int = 0) -> pd.Series:
    """
    Rolling majority (mode). On ties choose deterministic pseudo-random based on seed.
    Returns a Series aligned to input index.
    """
    rs = np.random.RandomState(seed)

    def mode_or_random(x):
        x = np.asarray(x)
        x = x[~pd.isna(x)]
        if x.size == 0:
            return np.nan
        vals, counts = np.unique(x, return_counts=True)
        maxc = counts.max()
        modes = vals[counts == maxc]
        if modes.size == 1:
            return modes[0]
        # deterministic tie-break
        return modes[rs.randint(0, modes.size)]

    out = series.rolling(window=window, min_periods=1).apply(lambda x: mode_or_random(x), raw=False)
    return out


def label_regime(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """
    Produce discrete market-regime labels:
      0 = flat/chop
      1 = bullish/trending-up
      2 = bearish/trending-down

    Signature: label_regime(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series

    Params allowed keys:
      atr_col="tc_atr", adx_col="tc_adx", adx_threshold=25,
      trend_strength_col="tc_trend_strength", trend_strength_threshold=0.5,
      vol_comp_col="tc_vol_comp_ratio", vote_window=5, seed=0, verbose=False
    """
    params = params or {}
    atr_col = str(params.get("atr_col", "tc_atr"))
    adx_col = str(params.get("adx_col", "tc_adx"))
    adx_threshold = float(params.get("adx_threshold", 25))
    trend_strength_col = str(params.get("trend_strength_col", "tc_trend_strength"))
    trend_strength_threshold = float(params.get("trend_strength_threshold", 0.5))
    vol_comp_col = str(params.get("vol_comp_col", "tc_vol_comp_ratio"))
    vote_window = int(params.get("vote_window", 5))
    seed = int(params.get("seed", 0))
    verbose = bool(params.get("verbose", False))

    # Defensive: ensure indexed
    df = _ensure_indexed(df).copy()
    # Normalize column names to strings to avoid numeric-key issues
    df.columns = df.columns.map(str)

    required = {
        atr_col: "average true range",
        adx_col: "ADX (trend strength)",
        trend_strength_col: "trend_strength indicator",
        vol_comp_col: "volume comparison ratio",
    }
    df = _align_features(df, required)

    # Ensure a safe close series
    if "close" not in df.columns:
        print("[regime] warning: 'close' not in df; creating synthetic close from ohlc if available.")
        if {"open", "high", "low"}.issubset(set(df.columns)):
            df["close"] = (pd.to_numeric(df["open"], errors="coerce") +
                           pd.to_numeric(df["high"], errors="coerce") +
                           pd.to_numeric(df["low"], errors="coerce")) / 3.0
        else:
            df["close"] = np.nan

    # numeric conversions with safe ffill/bfill
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    tc_atr = pd.to_numeric(df.get(atr_col, pd.Series(np.nan, index=df.index)), errors="coerce").ffill().bfill()
    tc_atr_norm = tc_atr / (close.replace(0, np.nan)).abs()
    adx = pd.to_numeric(df.get(adx_col, pd.Series(np.nan, index=df.index)), errors="coerce").ffill().bfill()
    trend_strength = pd.to_numeric(df.get(trend_strength_col, pd.Series(np.nan, index=df.index)), errors="coerce").ffill().bfill()

    # trending rule
    trending = (adx >= adx_threshold) & (trend_strength >= trend_strength_threshold)
    trending = trending.fillna(False)

    # direction via EMA; fallback to momentum sign
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    # compute a robust ATR quantile with fallback
    try:
        atr_q75 = float(tc_atr_norm.quantile(0.75))
    except Exception:
        atr_q75 = float(np.nanmedian(tc_atr_norm.values))

    direction_bull = (ema21 > ema50) & (tc_atr_norm < atr_q75)
    direction_bear = ~direction_bull

    raw = pd.Series(0, index=df.index, dtype=int)
    raw[(trending) & (direction_bull.fillna(False))] = 1
    raw[(trending) & (direction_bear.fillna(False))] = 2

    # Smooth with rolling majority
    smoothed = _rolling_majority(raw, max(1, vote_window), seed=seed)
    smoothed = smoothed.fillna(0).astype(int)
    smoothed.name = "regime_sig"

    # Unit checks
    if not isinstance(smoothed.index, pd.DatetimeIndex):
        smoothed.index = pd.to_datetime(smoothed.index)

    assert len(smoothed) == len(df)
    assert isinstance(smoothed.index, pd.DatetimeIndex)

    if verbose:
        print(f"[regime] counts: {dict(smoothed.value_counts().to_dict())}")
        for cls in [0, 1, 2]:
            idxs = smoothed[smoothed == cls].index[:3].tolist()
            print(f"[regime] sample idxs for {cls}: {idxs}")

    return smoothed
