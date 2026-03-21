# app/ml/labels/reversal.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional
from scipy.signal import argrelextrema

DEBUG_OUT = "out/labels_v02/reversal_debug.csv"


def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index(pd.DatetimeIndex(df.index))
        except Exception:
            if "datetime" in df.columns:
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["datetime"])))
            else:
                raise ValueError("Input df must have a DatetimeIndex or a 'datetime' column.")
    return df


def _align_features(df: pd.DataFrame, required_cols):
    for col in required_cols:
        if col not in df.columns:
            print(f"[reversal] warning: missing column '{col}'. Filling fallback NaN/0.")
            df[col] = 0 if str(col).startswith("ms_") else np.nan
    return df


def label_reversal(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series:
    """
    Identify short-term reversal points (binary 0/1).

    Signature:
        label_reversal(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series

    Params (subset):
      lookback=20
      roc_thresh=0.008
      cvd_divergence_win=5
      atr_mult=1.0
      rsi_thresh=(70,30)
      min_distance_minutes=3
      seed=0
      verbose=False
      mode="conservative"  # or "permissive"
    """
    params = params or {}
    lookback = int(params.get("lookback", 20))
    roc_thresh = float(params.get("roc_thresh", 0.008))
    cvd_win = int(params.get("cvd_divergence_win", 5))
    atr_mult = float(params.get("atr_mult", 1.0))
    rsi_thresh = params.get("rsi_thresh", (70, 30))
    min_distance_minutes = int(params.get("min_distance_minutes", 3))
    seed = int(params.get("seed", 0))
    verbose = bool(params.get("verbose", False))
    mode = str(params.get("mode", "conservative")).lower()

    # permissive testing adjustments
    if mode == "permissive":
        roc_thresh = float(params.get("roc_thresh", roc_thresh * 0.25))
        atr_mult = float(params.get("atr_mult", max(0.25, atr_mult * 0.5)))
        if hasattr(rsi_thresh, "__len__"):
            rsi_thresh = (rsi_thresh[0] - 5, rsi_thresh[1] + 5)

    df = _ensure_indexed(df).copy()

    # rename numeric or numeric-looking column labels to safe strings
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            rename_map[c] = f"unnamed_{str(c)}"
        else:
            if c.isdigit():
                rename_map[c] = f"unnamed_{c}"
    if rename_map:
        df = df.rename(columns=rename_map)
        if verbose:
            print(f"[reversal] info: renamed columns: {rename_map}")

    df.columns = df.columns.map(str)

    required_cols = [
        "close",
        "high",
        "low",
        "tc_atr",
        "tc_rsi_14",
        "ms_micro_divergence",
        "ms_sweep_flag",
        "ms_absorption_flag",
        "ms_aggression_idx",
        "ms_mid",
        "ms_spread",
        "ms_microtrend_vector",
        "ms_spread_z",
    ]
    df = _align_features(df, required_cols)

    # prepare core series
    close = pd.to_numeric(df["close"].ffill().bfill(), errors="coerce").astype(float)
    n = max(1, int(lookback / 4))
    try:
        highs_idx = argrelextrema(close.values, comparator=np.greater, order=n)[0]
        lows_idx = argrelextrema(close.values, comparator=np.less, order=n)[0]
    except Exception:
        highs_idx = np.array([], dtype=int)
        lows_idx = np.array([], dtype=int)

    # ---------- ms_mid proxy ----------
    if "ms_mid" in df.columns and df["ms_mid"].notna().any():
        ms_mid = pd.to_numeric(df["ms_mid"].ffill().bfill(), errors="coerce").astype(float)
    elif "ms_microtrend_vector" in df.columns and df["ms_microtrend_vector"].notna().any():
        vec = pd.to_numeric(df["ms_microtrend_vector"].fillna(0), errors="coerce").astype(float)
        # smoother proxy: small rolling mean then cumulative (gives trend)
        ms_mid = vec.rolling(5, min_periods=1).mean().cumsum().ffill().bfill()
    else:
        ms_mid = pd.Series(0.0, index=df.index)

    ms_mid = ms_mid.ffill().bfill().astype(float)

    # ---------- structural divergence heuristic (two tests) ----------
    window = max(3, cvd_win)
    ms_div = pd.Series(0, index=df.index, dtype=int)
    ms_mid_roll_max = ms_mid.rolling(window, min_periods=1).max()
    ms_mid_roll_min = ms_mid.rolling(window, min_periods=1).min()

    # 1) local-extrema mismatch (existing test)
    for idx in highs_idx:
        if idx < len(ms_mid):
            if ms_mid.iloc[idx] < ms_mid_roll_max.iloc[max(0, idx - window): idx + 1].max() - 1e-12:
                ms_div.iloc[idx] = 1
    for idx in lows_idx:
        if idx < len(ms_mid):
            if ms_mid.iloc[idx] > ms_mid_roll_min.iloc[max(0, idx - window): idx + 1].min() + 1e-12:
                ms_div.iloc[idx] = 1

    # 2) slope-based divergence: price slope significantly > ms_mid slope at extrema
    slope_window = max(2, int(window / 2))
    price_slope = (close - close.shift(slope_window)).fillna(0) / max(1, slope_window)
    ms_mid_slope = (ms_mid - ms_mid.shift(slope_window)).fillna(0) / max(1, slope_window)
    for idx in highs_idx:
        if idx < len(price_slope):
            ps = price_slope.iloc[idx]
            ms = ms_mid_slope.iloc[idx]
            # price making a stronger upward slope but ms_mid slope not matching
            if ps > 0 and ps > (abs(ms) * 1.2 + 1e-12) and ms <= ps * 0.6:
                ms_div.iloc[idx] = 1
    for idx in lows_idx:
        if idx < len(price_slope):
            ps = price_slope.iloc[idx]
            ms = ms_mid_slope.iloc[idx]
            if ps < 0 and abs(ps) > (abs(ms) * 1.2 + 1e-12) and abs(ms) <= abs(ps) * 0.6:
                ms_div.iloc[idx] = 1

    # prefer existing feature if present (union)
    try:
        existing_div = pd.to_numeric(df.get("ms_micro_divergence", pd.Series(0, index=df.index)).fillna(0), errors="coerce").fillna(0).astype(int)
        ms_div = ((existing_div > 0) | (ms_div > 0)).astype(int)
    except Exception:
        pass

    df["ms_micro_divergence"] = ms_div

    # ---------- aggression fallback ----------
    if ("ms_aggression_idx" not in df.columns) or pd.to_numeric(df["ms_aggression_idx"].fillna(0), errors="coerce").fillna(0).sum() == 0:
        if "ms_spread" in df.columns:
            spread = pd.to_numeric(df["ms_spread"].fillna(0), errors="coerce").abs()
            price_move = close.diff().abs().fillna(0)
            proxy = (spread * price_move).fillna(0)
            if proxy.sum() == 0:
                proxy = price_move.abs()
            p50 = float(np.percentile(proxy.values, 50)) if proxy.size else 0.0
            p75 = float(np.percentile(proxy.values, 75)) if proxy.size else p50
            denom = p50 if p50 > 1e-9 else (p75 if p75 > 1e-9 else (proxy.mean() if proxy.mean() > 1e-9 else 1.0))
            aggr_idx = (proxy / denom).fillna(0)
            df["ms_aggression_idx"] = aggr_idx
            if verbose:
                print(f"[reversal] info: ms_aggression_idx fallback created (p50={p50:.6f}, denom={denom:.6f})")
        else:
            df["ms_aggression_idx"] = 0.0
            if verbose:
                print("[reversal] info: ms_aggression_idx missing and ms_spread not available; using zeros fallback.")

    # ---------- numeric series ready ----------
    atr = pd.to_numeric(df.get("tc_atr", pd.Series(np.nan, index=df.index)).ffill().bfill(), errors="coerce").astype(float)
    rsi = pd.to_numeric(df.get("tc_rsi_14", pd.Series(np.nan, index=df.index)).ffill().bfill(), errors="coerce").astype(float)
    ms_div = pd.to_numeric(df["ms_micro_divergence"].fillna(0), errors="coerce").fillna(0).astype(int)
    ms_sweep = pd.to_numeric(df.get("ms_sweep_flag", pd.Series(0, index=df.index)).fillna(0), errors="coerce").fillna(0).astype(int)
    ms_abs = pd.to_numeric(df.get("ms_absorption_flag", pd.Series(0, index=df.index)).fillna(0), errors="coerce").fillna(0).astype(int)
    ms_aggr = pd.to_numeric(df.get("ms_aggression_idx", pd.Series(0, index=df.index)).fillna(0), errors="coerce").fillna(0).astype(float)
    roc = close.pct_change(periods=max(1, int(lookback / 2))).fillna(0)

    # prepare outputs & spacing
    reversal = pd.Series(0, index=df.index, dtype=int)
    last_flagged_ts = pd.Timestamp("1970-01-01", tz=df.index.tz)

    def set_flag(pos_idx: int) -> bool:
        nonlocal last_flagged_ts
        ts = df.index[pos_idx]
        if (ts - last_flagged_ts).total_seconds() < (min_distance_minutes * 60):
            return False
        reversal.iloc[pos_idx] = 1
        last_flagged_ts = ts
        return True

    rolling_med_atr = atr.rolling(lookback, min_periods=1).median().ffill().bfill()
    atr_safe = rolling_med_atr.copy()
    atr_safe[atr_safe <= 0] = np.nan

    flagged_highs = 0
    flagged_lows = 0

    # process extrema candidates (require div/sweep/abs + mom/roc + atr)
    for idx in highs_idx:
        cond_div = bool(ms_div.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_sweep = bool(ms_sweep.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_abs = bool(ms_abs.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_mom = (rsi.iloc[idx] >= rsi_thresh[0]) if hasattr(rsi_thresh, "__len__") else (rsi.iloc[idx] >= rsi_thresh)
        cond_roc = abs(roc.iloc[idx]) >= roc_thresh

        cond_atr = True
        if not np.isnan(atr_safe.iloc[idx]):
            cond_atr = atr.iloc[idx] >= (atr_mult * atr_safe.iloc[idx])

        if (cond_div or cond_sweep or cond_abs) and (cond_mom or cond_roc) and cond_atr:
            if set_flag(idx):
                flagged_highs += 1
        else:
            if verbose:
                reasons = []
                if not (cond_div or cond_sweep or cond_abs):
                    reasons.append("no_div/sweep/abs")
                if not (cond_mom or cond_roc):
                    reasons.append("no_mom/roc")
                if not cond_atr:
                    reasons.append("no_atr")
                print(f"[reversal] high idx {idx} suppressed: {','.join(reasons)}")

    for idx in lows_idx:
        cond_div = bool(ms_div.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_sweep = bool(ms_sweep.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_abs = bool(ms_abs.iloc[max(0, idx - cvd_win): idx + 1].sum() > 0)
        cond_mom = (rsi.iloc[idx] <= rsi_thresh[1]) if hasattr(rsi_thresh, "__len__") else (rsi.iloc[idx] <= rsi_thresh)
        cond_roc = abs(roc.iloc[idx]) >= roc_thresh

        cond_atr = True
        if not np.isnan(atr_safe.iloc[idx]):
            cond_atr = atr.iloc[idx] >= (atr_mult * atr_safe.iloc[idx])

        if (cond_div or cond_sweep or cond_abs) and (cond_mom or cond_roc) and cond_atr:
            if set_flag(idx):
                flagged_lows += 1
        else:
            if verbose:
                reasons = []
                if not (cond_div or cond_sweep or cond_abs):
                    reasons.append("no_div/sweep/abs")
                if not (cond_mom or cond_roc):
                    reasons.append("no_mom/roc")
                if not cond_atr:
                    reasons.append("no_atr")
                print(f"[reversal] low idx {idx} suppressed: {','.join(reasons)}")

    # ---------- aggression spikes: stricter acceptance ----------
    aggr_med = ms_aggr.rolling(lookback, min_periods=1).median().ffill().bfill()
    if aggr_med.sum() == 0:
        baseline = np.percentile(ms_aggr.values, 50) if ms_aggr.size else 0.0
        if baseline <= 0:
            baseline = np.percentile(ms_aggr.values, 75) if ms_aggr.size else 1.0
        aggr_med = pd.Series(baseline, index=ms_aggr.index)

    spike_thresh_mult = 1.5 if mode == "conservative" else 1.2
    aggr_spike_mask = ms_aggr > (aggr_med * spike_thresh_mult + 1e-12)
    spike_positions = np.where(aggr_spike_mask)[0]

    # widen ext_window a bit (helps accept spikes slightly off the exact extrema)
    ext_window = max(3, int(max(1, lookback * 0.5)))
    accepted_spikes = 0

    # prepare debug records
    debug_rows = []

    for pos in spike_positions:
        ts = df.index[pos]
        if (ts - last_flagged_ts).total_seconds() < (min_distance_minutes * 60):
            debug_rows.append({"pos": int(pos), "ts": ts, "reason": "suppressed_spacing"})
            continue

        # check for nearby divergence within ±cvd_win
        div_nearby = ms_div.iloc[max(0, pos - cvd_win): pos + cvd_win + 1].sum() > 0

        # check for nearby extrema within ±ext_window
        all_extrema = np.concatenate([highs_idx, lows_idx]) if (len(highs_idx) or len(lows_idx)) else np.array([], dtype=int)
        extrema_nearby = False
        if all_extrema.size:
            distances = np.abs(all_extrema - pos)
            extrema_nearby = np.any(distances <= ext_window)

        # alternative acceptance: strong local ROC + high aggression multiple
        roc_condition = abs(roc.iloc[pos]) >= (roc_thresh * 0.5)
        aggr_strength_cond = ms_aggr.iloc[pos] > (aggr_med.iloc[pos] * 2.5)  # require stronger spike

        accepted = False
        accept_reason = ""
        if div_nearby:
            accepted = True
            accept_reason = "div_nearby"
        elif extrema_nearby:
            accepted = True
            accept_reason = "ext_nearby"
        elif roc_condition and aggr_strength_cond:
            accepted = True
            accept_reason = "roc+aggr"
        else:
            accept_reason = "no_div_ext_weak_roc"

        if accepted:
            reversal.iloc[pos] = 1
            last_flagged_ts = ts
            accepted_spikes += 1
            debug_rows.append({"pos": int(pos), "ts": ts, "reason": accept_reason, "ms_aggr": float(ms_aggr.iloc[pos]), "aggr_med": float(aggr_med.iloc[pos]), "roc": float(roc.iloc[pos])})
        else:
            debug_rows.append({"pos": int(pos), "ts": ts, "reason": accept_reason, "ms_aggr": float(ms_aggr.iloc[pos]), "aggr_med": float(aggr_med.iloc[pos]), "roc": float(roc.iloc[pos])})
            if verbose:
                print(f"[reversal] aggr spike at pos {pos} ignored ({accept_reason})")

    # persist debug CSV for tuning
    try:
        if debug_rows:
            os.makedirs(os.path.dirname(DEBUG_OUT), exist_ok=True)
            pd.DataFrame(debug_rows).to_csv(DEBUG_OUT, index=False)
            if verbose:
                print(f"[reversal] debug CSV written to {DEBUG_OUT} (rows={len(debug_rows)})")
    except Exception:
        if verbose:
            print("[reversal] warning: failed to write debug CSV")

    if verbose:
        print(f"[reversal] flagged_highs={flagged_highs}, flagged_lows={flagged_lows}, aggr_spikes_total={len(spike_positions)}, aggr_spikes_accepted={accepted_spikes}, total={int(reversal.sum())}")

    reversal.name = "reversal_sig"

    assert isinstance(reversal.index, pd.DatetimeIndex)
    assert len(reversal) == len(df)

    return reversal
