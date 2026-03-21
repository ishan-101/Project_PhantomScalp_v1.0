# app/ml/labels/cycle.py
"""
app/ml/labels/cycle.py

Robust cycle phase labeler (discrete and phase modes).

Discrete mapping:
  0 = bottom
  1 = rise
  2 = top
  3 = fall

API:
  label_cycle(df: pd.DataFrame, params: Optional[dict] = None) -> pd.Series
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks


def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    if "datetime" in df.columns:
        try:
            tmp = df.copy()
            tmp["datetime"] = pd.to_datetime(tmp["datetime"])
            tmp = tmp.set_index("datetime")
            return tmp
        except Exception:
            pass
    try:
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        return tmp
    except Exception:
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(df), freq="T")
        tmp = df.copy()
        tmp.index = idx
        return tmp


def _phase_to_discrete(phase: np.ndarray) -> np.ndarray:
    labels = np.full(len(phase), 1, dtype=int)
    labels[(phase >= -np.pi) & (phase < -np.pi / 2)] = 0
    labels[(phase >= -np.pi / 2) & (phase < 0)] = 1
    labels[(phase >= 0) & (phase < np.pi / 2)] = 2
    labels[(phase >= np.pi / 2) & (phase <= np.pi)] = 3
    return labels


def _try_relaxed_peaks(arr: np.ndarray, peak_trough_window: int, std: float, prominence_mult: float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Try progressively relaxed peak/trough detection. Accepts a prominence multiplier to
    make the detection more/less conservative.
    """
    # distances relative to requested peak_trough_window (larger -> stricter)
    distances = [max(1, int(peak_trough_window / d)) for d in (3, 4, 6, 10)]
    # progressively relax prominence
    prominence_mults = [prominence_mult, prominence_mult * 0.5, prominence_mult * 0.2, prominence_mult * 0.05, 0.0]
    for dist in distances:
        for pm in prominence_mults:
            prom = max(1e-12, std * pm)
            try:
                peaks, _ = find_peaks(arr, distance=dist, prominence=prom)
                troughs, _ = find_peaks(-arr, distance=dist, prominence=prom)
            except Exception:
                peaks = np.array([], dtype=int)
                troughs = np.array([], dtype=int)
            if len(peaks) >= 2 and len(troughs) >= 2:
                return peaks.tolist(), troughs.tolist()
    # final permissive attempt
    try:
        peaks, _ = find_peaks(arr, distance=1, prominence=max(1e-12, std * (prominence_mult * 0.05)))
        troughs, _ = find_peaks(-arr, distance=1, prominence=max(1e-12, std * (prominence_mult * 0.05)))
        return peaks.tolist(), troughs.tolist()
    except Exception:
        return [], []


def label_cycle(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Params (subset):
      - method: "discrete" (default) or "phase"
      - peak_trough_window: int (default 30)
      - filter_window: int smoothing window (default 5)
      - prominence_mult: float (default 0.1)   # tuning knob
      - seed: int (default 0)
      - verbose: bool (default False)
      - debug_peaks_out: optional path to write peaks/trough debug CSV
    """
    params = params or {}
    method = params.get("method", "discrete")
    peak_trough_window = int(params.get("peak_trough_window", 30))
    filter_window = int(params.get("filter_window", 5))
    prominence_mult = float(params.get("prominence_mult", 0.1))
    seed = int(params.get("seed", 0))
    verbose = bool(params.get("verbose", False))
    debug_peaks_out = params.get("debug_peaks_out", None)

    df_in = _ensure_indexed(df)
    idx = df_in.index

    if "close" not in df_in.columns:
        if verbose:
            print("[cycle] warning: 'close' column missing; returning fallback series.")
        if method == "phase":
            return pd.Series(np.nan, index=idx, name="cycle_sig", dtype=float)
        return pd.Series(0, index=idx, name="cycle_sig", dtype=int)

    close = pd.to_numeric(df_in["close"], errors="coerce")
    nan_mask = close.isna()
    close_filled = close.ffill().bfill()

    if filter_window > 1:
        s_smooth = close_filled.rolling(window=filter_window, min_periods=1).mean()
    else:
        s_smooth = close_filled

    arr = s_smooth.values.astype(float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        if method == "phase":
            return pd.Series(np.nan, index=idx, name="cycle_sig", dtype=float)
        return pd.Series(0, index=idx, name="cycle_sig", dtype=int)

    # PHASE mode: continuous analytic phase
    if method == "phase":
        try:
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            analytic = hilbert(np.nan_to_num(detr))
            phase = np.angle(analytic)
            phase_series = pd.Series(phase, index=idx, name="cycle_sig", dtype=float)
            phase_series[nan_mask] = np.nan
            if verbose:
                print(f"[cycle][phase] produced phase series len={len(phase_series)}, nan_count={int(phase_series.isna().sum())}")
            return phase_series
        except Exception as e:
            if verbose:
                print("[cycle][phase] hilbert failed; fallback to normalized detrend:", e)
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            norm = detr / (np.nanstd(detr) + 1e-9)
            phase_series = pd.Series(np.arctan(norm), index=idx, name="cycle_sig", dtype=float)
            phase_series[nan_mask] = np.nan
            return phase_series

    # DISCRETE mode
    std = float(np.nanstd(arr))
    if std <= 0 or np.isnan(std):
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(0.0, 1e-9, size=arr.shape)
        std = 1e-9

    peaks, troughs = _try_relaxed_peaks(arr, peak_trough_window, std, prominence_mult=prominence_mult)

    labels = np.full(len(arr), 1, dtype=int)  # default = rise
    slope = np.gradient(arr)
    labels[slope < 0] = 3  # fall

    # try to mark extremes if peaks/troughs found
    if len(peaks) >= 2 and len(troughs) >= 2:
        half_w = max(1, filter_window // 2)
        for p in peaks:
            left = max(0, p - half_w)
            right = min(len(labels) - 1, p + half_w)
            labels[left: right + 1] = 2
        for t in troughs:
            left = max(0, t - half_w)
            right = min(len(labels) - 1, t + half_w)
            labels[left: right + 1] = 0
    else:
        # fallback: analytic-phase-based segmentation
        try:
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            analytic = hilbert(np.nan_to_num(detr))
            phase = np.angle(analytic)
            labels = _phase_to_discrete(phase)
            if verbose:
                print("[cycle] peak/trough detection failed — used phase->discrete fallback.")
        except Exception:
            q_low = np.nanpercentile(slope, 33)
            q_high = np.nanpercentile(slope, 66)
            labels = np.where(slope <= q_low, 3, np.where(slope >= q_high, 1, 1)).astype(int)
            try:
                local_max = (s_smooth == s_smooth.rolling(window=max(3, filter_window), center=True, min_periods=1).max())
                local_min = (s_smooth == s_smooth.rolling(window=max(3, filter_window), center=True, min_periods=1).min())
                labels[np.where(local_max.fillna(False).values)[0]] = 2
                labels[np.where(local_min.fillna(False).values)[0]] = 0
            except Exception:
                pass
            if verbose:
                print("[cycle] used slope-quantile fallback.")

    out = pd.Series(labels, index=idx, dtype=int, name="cycle_sig")
    out[nan_mask] = 0

    # degenerate safety: if only one unique label, force phase mapping
    uniques = pd.Series(out.values).unique()
    if len(uniques) == 1:
        try:
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            analytic = hilbert(np.nan_to_num(detr))
            phase = np.angle(analytic)
            forced = _phase_to_discrete(phase)
            out = pd.Series(forced, index=idx, dtype=int, name="cycle_sig")
            out[nan_mask] = 0
            if verbose:
                print("[cycle] forced phase->discrete because labels were degenerate.")
        except Exception:
            pass

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    try:
        assert len(out) == len(df_in)
    except AssertionError:
        out = out.reindex(df_in.index, fill_value=0)

    if verbose:
        vals = dict(pd.Series(out.values).value_counts().to_dict())
        print(f"[cycle][discrete] final counts: {vals}")

    # optional debug: write peaks/troughs list for inspection
    if debug_peaks_out and verbose:
        try:
            rows = []
            for p in peaks:
                rows.append({"pos": int(p), "ts": str(idx[p]), "kind": "peak", "value": float(arr[p])})
            for t in troughs:
                rows.append({"pos": int(t), "ts": str(idx[t]), "kind": "trough", "value": float(arr[t])})
            os.makedirs(os.path.dirname(debug_peaks_out), exist_ok=True)
            pd.DataFrame(rows).to_csv(debug_peaks_out, index=False)
            if verbose:
                print(f"[cycle] wrote peaks/troughs debug to {debug_peaks_out}")
        except Exception as e:
            if verbose:
                print("[cycle] warning: failed to write peaks debug:", e)

    return out
