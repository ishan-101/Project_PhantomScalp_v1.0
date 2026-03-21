# app/features/time_cycle.py
from __future__ import annotations
"""
time_cycle.py - extended (v0.2, full + extras)

Public:
    compute_time_cycle_features(df: pd.DataFrame, params: Optional[dict]=None) -> pd.DataFrame

Added: SMA20, Trend Strength Index (TSI), Volatility Compression Ratio,
Session volatility score, exchange micro-session participation, explicit tc_sma_20,
tc_vol_comp_ratio, tc_trend_strength.
"""
from typing import Optional, Dict, Sequence
import numpy as np
import pandas as pd

FEATURE_MANIFEST: Dict[str, str] = {
    "ema": "tc_ema_{}",
    "sma": "tc_sma_{}",
    "vwap": "tc_vwap",
    "atr": "tc_atr",
    "atr_norm": "tc_atr_norm",
    "atr_band_upper": "tc_atr_band_upper",
    "atr_band_lower": "tc_atr_band_lower",
    "rsi": "tc_rsi_{}",
    "macd": "tc_macd",
    "macd_sig": "tc_macd_sig",
    "macd_hist": "tc_macd_hist",
    "bb_up": "tc_bb_upper",
    "bb_mid": "tc_bb_mid",
    "bb_low": "tc_bb_lower",
    "kc_up": "tc_kc_upper",
    "kc_low": "tc_kc_low",
    "don_high": "tc_don_high",
    "don_low": "tc_don_low",
    "sto_k": "tc_sto_k",
    "sto_d": "tc_sto_d",
    "willr": "tc_willr",
    "roc": "tc_roc",
    "mom": "tc_mom",
    "adx": "tc_adx",
    "di_plus": "tc_di_plus",
    "di_minus": "tc_di_minus",
    "chop": "tc_chop",
    "squeeze_on": "tc_squeeze_on",
    "hour": "tc_hour",
    "minute": "tc_minute",
    "session": "tc_session",
    "hv": "tc_hv_{}",
    "cvi": "tc_cvi_{}_{}",
    "hma": "tc_hma_{}",
    "wma": "tc_wma_{}",
    "dema": "tc_dema_{}",
    "tema": "tc_tema_{}",
    "supertrend_dir": "tc_supertrend_dir",
    "vol_comp": "tc_vol_comp_ratio",
    "trend_strength": "tc_trend_strength",
    "session_vol": "tc_session_vol_score",
    "session_part": "tc_session_participation",
    "micro_session_flag": "tc_micro_session_flag",
}

def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column")
    return df

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()

def _wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    def _roll(x):
        v = np.asarray(x, dtype=float)
        w = weights[-len(v):]
        return np.dot(v, w) / w.sum()
    return series.rolling(length, min_periods=1).apply(lambda x: _roll(x), raw=False)

def _dema(series: pd.Series, length: int) -> pd.Series:
    e1 = _ema(series, length)
    e2 = _ema(e1, length)
    return 2 * e1 - e2

def _tema(series: pd.Series, length: int) -> pd.Series:
    e1 = _ema(series, length)
    e2 = _ema(e1, length)
    e3 = _ema(e2, length)
    return 3 * (e1 - e2) + e3

def _hma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, int(length / 2))
    sq = max(1, int(np.sqrt(length)))
    return _wma(2 * _wma(series, half) - _wma(series, length), sq)

def _tr(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = _tr(df)
    return tr.rolling(length, min_periods=1).mean().rename(FEATURE_MANIFEST["atr"])

def atr_norm(df: pd.DataFrame, length: int = 14) -> pd.Series:
    a = atr(df, length)
    price = df["close"].astype(float)
    return (a / price.replace(0, np.nan)).rename(FEATURE_MANIFEST["atr_norm"])

def atr_bands(df: pd.DataFrame, length: int = 14, mult: float = 2.0):
    a = atr(df, length)
    mid = df["close"].astype(float).rolling(length, min_periods=1).mean()
    upper = (mid + mult * a).rename(FEATURE_MANIFEST["atr_band_upper"])
    lower = (mid - mult * a).rename(FEATURE_MANIFEST["atr_band_lower"])
    return upper, lower

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"].astype(float) * df["volume"].astype(float)).cumsum()
    v = df["volume"].astype(float).cumsum().replace(0, np.nan)
    return (pv / v).rename(FEATURE_MANIFEST["vwap"])

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.rename(FEATURE_MANIFEST["rsi"].format(length))

def macd(series: pd.Series, fast=12, slow=26, sig=9):
    fast_e = _ema(series, fast)
    slow_e = _ema(series, slow)
    macd_line = fast_e - slow_e
    sig_line = _ema(macd_line, sig)
    hist = macd_line - sig_line
    return macd_line.rename(FEATURE_MANIFEST["macd"]), sig_line.rename(FEATURE_MANIFEST["macd_sig"]), hist.rename(FEATURE_MANIFEST["macd_hist"])

def bollinger(series: pd.Series, length=20, mult=2.0):
    mid = _sma(series, length)
    std = series.rolling(length, min_periods=1).std()
    up = (mid + mult * std).rename(FEATURE_MANIFEST["bb_up"])
    low = (mid - mult * std).rename(FEATURE_MANIFEST["bb_low"])
    return up, mid.rename(FEATURE_MANIFEST["bb_mid"]), low

def keltner(df: pd.DataFrame, length=20, mult=1.5):
    mid = _ema(df["close"].astype(float), length)
    a = atr(df, length)
    up = (mid + mult * a).rename(FEATURE_MANIFEST["kc_up"])
    low = (mid - mult * a).rename(FEATURE_MANIFEST["kc_low"])
    return up, low

def donchian(df: pd.DataFrame, length=20):
    high = df["high"].astype(float).rolling(length, min_periods=1).max().rename(FEATURE_MANIFEST["don_high"])
    low = df["low"].astype(float).rolling(length, min_periods=1).min().rename(FEATURE_MANIFEST["don_low"])
    return high, low

def stochastic(df: pd.DataFrame, k_window=14, d_window=3):
    low = df["low"].astype(float).rolling(k_window, min_periods=1).min()
    high = df["high"].astype(float).rolling(k_window, min_periods=1).max()
    k = ((df["close"].astype(float) - low) / (high - low).replace(0, np.nan) * 100).rename(FEATURE_MANIFEST["sto_k"])
    d = k.rolling(d_window, min_periods=1).mean().rename(FEATURE_MANIFEST["sto_d"])
    return k, d

def williams_r(df: pd.DataFrame, length=14):
    highest = df["high"].astype(float).rolling(length, min_periods=1).max()
    lowest = df["low"].astype(float).rolling(length, min_periods=1).min()
    wr = ((highest - df["close"].astype(float)) / (highest - lowest).replace(0, np.nan) * -100).rename(FEATURE_MANIFEST["willr"])
    return wr

def roc(series: pd.Series, length=12):
    return ((series - series.shift(length)) / series.shift(length).replace(0, np.nan)).rename(FEATURE_MANIFEST["roc"])

def mom(series: pd.Series, length=10):
    return (series - series.shift(length)).rename(FEATURE_MANIFEST["mom"])

def adx(df: pd.DataFrame, length=14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _tr(df)
    atr_series = tr.rolling(length, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(length, min_periods=1).sum() / atr_series.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(length, min_periods=1).sum() / atr_series.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_line = dx.rolling(length, min_periods=1).mean()
    return adx_line.rename(FEATURE_MANIFEST["adx"]), plus_di.rename(FEATURE_MANIFEST["di_plus"]), minus_di.rename(FEATURE_MANIFEST["di_minus"])

def choppiness_index(df: pd.DataFrame, length=14):
    tr_sum = _tr(df).rolling(length, min_periods=1).sum()
    hi_low = (df["high"].astype(float).rolling(length, min_periods=1).max() - df["low"].astype(float).rolling(length, min_periods=1).min())
    chop = 100 * np.log10(tr_sum.replace(0, np.nan) / hi_low.replace(0, np.nan)) / np.log10(length)
    return chop.rename(FEATURE_MANIFEST["chop"])

def squeeze_on(df: pd.DataFrame, bb_len=20, kc_len=20, bb_mult=2.0, kc_mult=1.5):
    close = df["close"].astype(float)
    bb_up, bb_mid, bb_low = bollinger(close, bb_len, bb_mult)
    kc_up, kc_low = keltner(df, kc_len, kc_mult)
    squeeze = ( (bb_up - bb_low) < (kc_up - kc_low) ).astype(int).rename(FEATURE_MANIFEST["squeeze_on"])
    return squeeze

def trend_strength_index(df: pd.DataFrame, ema_fast=9, ema_slow=21):
    # Simple composite TSI: normalized EMA slope magnitude * ADX scaled to 0-1
    close = df["close"].astype(float)
    ema_f = _ema(close, ema_fast)
    ema_s = _ema(close, ema_slow)
    ema_diff = (ema_f - ema_s).abs()
    ema_diff_norm = (ema_diff / close.replace(0, np.nan)).rolling(ema_slow, min_periods=1).mean()
    adx_line, _, _ = adx(df)
    # scale both into 0..1
    a_norm = (adx_line / 50.0).clip(0,1)
    e_norm = (ema_diff_norm / (ema_diff_norm.rolling(ema_slow, min_periods=1).max().replace(0, np.nan))).fillna(0)
    tsi = (a_norm * 0.6 + e_norm * 0.4).rename(FEATURE_MANIFEST["trend_strength"])
    return tsi

def volatility_compression_ratio(df: pd.DataFrame, bb_len=20, kc_len=20, bb_mult=2.0, kc_mult=1.5):
    close = df["close"].astype(float)
    bb_up, bb_mid, bb_low = bollinger(close, bb_len, bb_mult)
    kc_up, kc_low = keltner(df, kc_len, 1.5)
    bb_width = (bb_up - bb_low).abs()
    kc_width = (kc_up - kc_low).abs().replace(0, np.nan)
    ratio = (bb_width / kc_width).rename(FEATURE_MANIFEST["vol_comp"])
    return ratio

def session_volatility_score(df: pd.DataFrame, window=30):
    # per-session HV z-score: compute HV per session bucket (tc_session) then zscore
    hv30 = ((np.log(df["close"].astype(float)).diff()).rolling(window, min_periods=1).std() * np.sqrt(window))
    session = session_label(df.index)
    sess_mean = hv30.groupby(session).transform("mean")
    sess_std = hv30.groupby(session).transform("std").replace(0, np.nan)
    score = ((hv30 - sess_mean) / sess_std).rename(FEATURE_MANIFEST["session_vol"])
    return score.fillna(0)

def session_label(index: pd.DatetimeIndex):
    hours = index.hour
    session = pd.Series(0, index=index, name=FEATURE_MANIFEST["session"])
    session[(hours >= 0) & (hours < 7)] = 1   # Asia
    session[(hours >= 7) & (hours < 13)] = 2  # EU
    session[(hours >= 13) & (hours < 24)] = 3 # US
    return session

def micro_session_participation(df: pd.DataFrame):
    # compute relative volume participation per session normalized
    session = session_label(df.index)
    vol = df["volume"].astype(float)
    sess_total = vol.groupby(session).transform("sum").replace(0, np.nan)
    part = (vol / sess_total).rename(FEATURE_MANIFEST["session_part"])
    # flag micro-session anomaly if participation > historical mean + 1.5*std
    mean = part.rolling(1440, min_periods=1).mean()
    std = part.rolling(1440, min_periods=1).std().replace(0, np.nan)
    flag = ((part - mean) > 1.5 * std).astype(int).rename(FEATURE_MANIFEST["micro_session_flag"])
    return part.fillna(0), flag.fillna(0)

def historical_volatility(df: pd.DataFrame, window: int = 30):
    logret = np.log(df["close"].astype(float)).diff()
    hv = logret.rolling(window, min_periods=1).std() * np.sqrt(window)
    return hv.rename(FEATURE_MANIFEST["hv"].format(window))

def chaikin_volatility(df: pd.DataFrame, short: int = 10, long: int = 20):
    hl = (df["high"] - df["low"]).astype(float)
    s = hl.ewm(span=short, adjust=False).mean()
    l = hl.ewm(span=long, adjust=False).mean()
    return ((s - l) / l.replace(0, np.nan)).rename(FEATURE_MANIFEST["cvi"].format(short, long))

def supertrend(df: pd.DataFrame, atr_len: int = 10, multiplier: float = 3.0):
    # preserve earlier supertrend dir method but return only direction series
    price = df["close"].astype(float)
    a = atr(df, atr_len)
    hl2 = (df["high"].astype(float) + df["low"].astype(float)) / 2.0
    upper = (hl2 + multiplier * a)
    lower = (hl2 - multiplier * a)
    dir_ser = pd.Series(0, index=df.index, name=FEATURE_MANIFEST["supertrend_dir"])
    prev_dir = 1
    prev_st = price.iloc[0] if len(price) else np.nan
    st = pd.Series(np.nan, index=df.index)
    for i, idx in enumerate(df.index):
        up = upper.loc[idx]
        low = lower.loc[idx]
        p = price.loc[idx]
        if i == 0:
            prev_st = up
            dir_ser.iloc[i] = 1
            st.iloc[i] = prev_st
            continue
        curr_dir = 1 if p > prev_st else -1 if p < prev_st else prev_dir
        dir_ser.iloc[i] = curr_dir
        prev_dir = curr_dir
        prev_st = up if curr_dir == -1 else low if curr_dir == 1 else prev_st
    return dir_ser

def compute_time_cycle_features(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    params = params or {}
    df = _ensure_indexed(df)
    close = df["close"].astype(float)

    parts = {}

    # EMAs and SMAs (include SMA20 explicitly)
    for L in (9, 21, 50, 200):
        parts[FEATURE_MANIFEST["ema"].format(L)] = _ema(close, L)
        parts[FEATURE_MANIFEST["sma"].format(L)] = _sma(close, L)
    # ensure SMA20 present
    parts[FEATURE_MANIFEST["sma"].format(20)] = _sma(close, 20)

    # VWAP
    try:
        parts[FEATURE_MANIFEST["vwap"]] = vwap(df)
    except Exception:
        parts[FEATURE_MANIFEST["vwap"]] = pd.Series(np.nan, index=df.index)

    # ATR & bands
    parts[FEATURE_MANIFEST["atr"]] = atr(df)
    parts[FEATURE_MANIFEST["atr_norm"]] = atr_norm(df)
    ub, lb = atr_bands(df)
    parts[FEATURE_MANIFEST["atr_band_upper"]] = ub
    parts[FEATURE_MANIFEST["atr_band_lower"]] = lb

    # RSI
    parts[FEATURE_MANIFEST["rsi"].format(14)] = rsi(close, 14)

    # MACD
    macd_line, macd_sig, macd_hist = macd(close)
    parts[FEATURE_MANIFEST["macd"]] = macd_line
    parts[FEATURE_MANIFEST["macd_sig"]] = macd_sig
    parts[FEATURE_MANIFEST["macd_hist"]] = macd_hist

    # Bollinger / Keltner / Donchian
    bb_up, bb_mid, bb_low = bollinger(close)
    parts[FEATURE_MANIFEST["bb_up"]] = bb_up
    parts[FEATURE_MANIFEST["bb_mid"]] = bb_mid
    parts[FEATURE_MANIFEST["bb_low"]] = bb_low
    kc_up, kc_low = keltner(df)
    parts[FEATURE_MANIFEST["kc_up"]] = kc_up
    parts[FEATURE_MANIFEST["kc_low"]] = kc_low
    don_h, don_l = donchian(df)
    parts[FEATURE_MANIFEST["don_high"]] = don_h
    parts[FEATURE_MANIFEST["don_low"]] = don_l

    # Stochastic / Williams / ROC / MOM
    sto_k, sto_d = stochastic(df)
    parts[FEATURE_MANIFEST["sto_k"]] = sto_k
    parts[FEATURE_MANIFEST["sto_d"]] = sto_d
    parts[FEATURE_MANIFEST["willr"]] = williams_r(df)
    parts[FEATURE_MANIFEST["roc"]] = roc(close)
    parts[FEATURE_MANIFEST["mom"]] = mom(close)

    # ADX / DI
    parts[FEATURE_MANIFEST["adx"]], parts[FEATURE_MANIFEST["di_plus"]], parts[FEATURE_MANIFEST["di_minus"]] = adx(df)

    # Choppiness
    parts[FEATURE_MANIFEST["chop"]] = choppiness_index(df)

    # Squeeze
    parts[FEATURE_MANIFEST["squeeze_on"]] = squeeze_on(df)

    # Trend Strength Index
    parts[FEATURE_MANIFEST["trend_strength"]] = trend_strength_index(df)

    # Volatility compression ratio
    parts[FEATURE_MANIFEST["vol_comp"]] = volatility_compression_ratio(df)

    # Hour/minute/session
    parts[FEATURE_MANIFEST["hour"]] = pd.Series(df.index.hour, index=df.index, name=FEATURE_MANIFEST["hour"])
    parts[FEATURE_MANIFEST["minute"]] = pd.Series(df.index.minute, index=df.index, name=FEATURE_MANIFEST["minute"])
    parts[FEATURE_MANIFEST["session"]] = session_label(df.index)

    # Session volatility and participation
    parts[FEATURE_MANIFEST["session_vol"]] = session_volatility_score(df)
    part_series, flag_series = micro_session_participation(df)
    parts[FEATURE_MANIFEST["session_part"]] = part_series
    parts[FEATURE_MANIFEST["micro_session_flag"]] = flag_series

    # HV/CVI
    parts[FEATURE_MANIFEST["hv"].format(30)] = historical_volatility(df, 30)
    parts[FEATURE_MANIFEST["hv"].format(90)] = historical_volatility(df, 90)
    parts[FEATURE_MANIFEST["cvi"].format(10, 20)] = chaikin_volatility(df, 10, 20)

    # HMA/WMA/DEMA/TEMA
    parts[FEATURE_MANIFEST["hma"].format(21)] = _hma(close, 21)
    parts[FEATURE_MANIFEST["wma"].format(21)] = _wma(close, 21)
    parts[FEATURE_MANIFEST["dema"].format(21)] = _dema(close, 21)
    parts[FEATURE_MANIFEST["tema"].format(21)] = _tema(close, 21)

    # Supertrend dir
    try:
        parts[FEATURE_MANIFEST["supertrend_dir"]] = supertrend(df)
    except Exception:
        parts[FEATURE_MANIFEST["supertrend_dir"]] = pd.Series(0, index=df.index)

    out = pd.concat([s.rename(k) if isinstance(s, pd.Series) else s for k, s in parts.items()], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    print("[tc] computed features:", list(out.columns)[:80])
    return out
