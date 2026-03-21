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

Notes:
 - Defensive: never raises on missing columns; returns fallback series when needed.
 - Index-preserving: output indexed by input DatetimeIndex (or converted).
 - Timezone-safe: preserves tz-aware indexes.
 - Deterministic when 'seed' provided.
 - Verbose via params["verbose"] prints diagnostics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks


def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by a DatetimeIndex. Create sensible fallback index if necessary."""
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
    # try coerce index to datetime
    try:
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        return tmp
    except Exception:
        # final fallback: generate a minute-based DatetimeIndex ending now (UTC)
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(df), freq="T")
        tmp = df.copy()
        tmp.index = idx
        return tmp


def _phase_to_discrete(phase: np.ndarray) -> np.ndarray:
    """
    Map continuous phase in [-pi, pi] to discrete labels:
     -pi .. -pi/2 -> bottom (0)
     -pi/2 .. 0  -> rise   (1)
      0  .. +pi/2 -> top    (2)
    +pi/2 .. pi  -> fall   (3)
    """
    labels = np.full(len(phase), 1, dtype=int)
    labels[(phase >= -np.pi) & (phase < -np.pi / 2)] = 0
    labels[(phase >= -np.pi / 2) & (phase < 0)] = 1
    labels[(phase >= 0) & (phase < np.pi / 2)] = 2
    labels[(phase >= np.pi / 2) & (phase <= np.pi)] = 3
    return labels


def _try_relaxed_peaks(arr: np.ndarray, peak_trough_window: int, std: float) -> Tuple[List[int], List[int]]:
    """
    Attempt multiple relaxations of prominence and distance to find peaks/troughs.
    Returns lists of peak indices and trough indices (may be empty).
    """
    distances = [max(1, int(peak_trough_window / d)) for d in (3, 4, 6, 10)]  # progressively smaller distances
    prominence_mults = [0.1, 0.05, 0.02, 0.0]  # progressively easier prominence
    for dist in distances:
        for pm in prominence_mults:
            prom = max(1e-9, std * pm)
            try:
                peaks, _ = find_peaks(arr, distance=dist, prominence=prom)
                troughs, _ = find_peaks(-arr, distance=dist, prominence=prom)
            except Exception:
                peaks = np.array([], dtype=int)
                troughs = np.array([], dtype=int)
            # require at least 2 peaks and 2 troughs for decent segmentation
            if len(peaks) >= 2 and len(troughs) >= 2:
                return peaks.tolist(), troughs.tolist()
    # final attempt: very permissive (distance=1, tiny prominence)
    try:
        peaks, _ = find_peaks(arr, distance=1, prominence=max(1e-9, std * 0.005))
        troughs, _ = find_peaks(-arr, distance=1, prominence=max(1e-9, std * 0.005))
        return peaks.tolist(), troughs.tolist()
    except Exception:
        return [], []


def label_cycle(df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    Main entrypoint.

    Params:
      - method: "discrete" (default) or "phase"
      - peak_trough_window: int (default 30)
      - filter_window: int smoothing window (default 5)
      - prominence_mult: float base multiplier for prominence (default 0.1) [used as initial value]
      - seed: int for deterministic small-noise in degenerate cases
      - verbose: bool
    """
    params = params or {}
    method = params.get("method", "discrete")
    peak_trough_window = int(params.get("peak_trough_window", 30))
    filter_window = int(params.get("filter_window", 5))
    prominence_mult = float(params.get("prominence_mult", 0.1))
    seed = int(params.get("seed", 0))
    verbose = bool(params.get("verbose", False))

    # Ensure indexed and preserve copy
    df_in = _ensure_indexed(df)
    idx = df_in.index

    # If close missing -> fallback series of zeros or NaNs
    if "close" not in df_in.columns:
        if verbose:
            print("[cycle] warning: 'close' column missing; returning fallback series.")
        if method == "phase":
            return pd.Series(np.nan, index=idx, name="cycle_sig", dtype=float)
        return pd.Series(0, index=idx, name="cycle_sig", dtype=int)

    # canonical close
    close = pd.to_numeric(df_in["close"], errors="coerce")
    nan_mask = close.isna()
    # prepare filled series for analysis
    close_filled = close.ffill().bfill()

    # smoothing
    if filter_window > 1:
        s_smooth = close_filled.rolling(window=filter_window, min_periods=1).mean()
    else:
        s_smooth = close_filled

    arr = s_smooth.values.astype(float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        # empty fallback
        if method == "phase":
            return pd.Series(np.nan, index=idx, name="cycle_sig", dtype=float)
        return pd.Series(0, index=idx, name="cycle_sig", dtype=int)

    # PHASE mode (continuous)
    if method == "phase":
        try:
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            analytic = hilbert(np.nan_to_num(detr))
            phase = np.angle(analytic)
            phase_series = pd.Series(phase, index=idx, name="cycle_sig", dtype=float)
            phase_series[nan_mask] = np.nan
            assert isinstance(phase_series.index, pd.DatetimeIndex)
            assert len(phase_series) == len(df_in)
            if verbose:
                print(f"[cycle][phase] produced phase series len={len(phase_series)}, nan_count={int(phase_series.isna().sum())}")
            return phase_series
        except Exception as e:
            if verbose:
                print("[cycle][phase] hilbert failed, fallback to normalized detrend:", e)
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            norm = detr / (np.nanstd(detr) + 1e-9)
            phase_series = pd.Series(np.arctan(norm), index=idx, name="cycle_sig", dtype=float)
            phase_series[nan_mask] = np.nan
            return phase_series

    # DISCRETE mode
    # compute data scale
    std = float(np.nanstd(arr))
    if std <= 0 or np.isnan(std):
        # degenerate: add tiny deterministic noise to make find_peaks tolerant
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(0.0, 1e-9, size=arr.shape)
        std = 1e-9

    # initial find_peaks attempt (relaxed attempts inside helper)
    peaks, troughs = _try_relaxed_peaks(arr, peak_trough_window, std)

    # If we found enough peaks/troughs, construct labels by marking windows around extrema
    labels = np.full(len(arr), 1, dtype=int)  # default = rise
    slope = np.gradient(arr)
    labels[slope < 0] = 3  # fall

    if len(peaks) >= 2 and len(troughs) >= 2:
        half_w = max(1, filter_window // 2)
        for p in peaks:
            left = max(0, p - half_w)
            right = min(len(labels) - 1, p + half_w)
            labels[left : right + 1] = 2
        for t in troughs:
            left = max(0, t - half_w)
            right = min(len(labels) - 1, t + half_w)
            labels[left : right + 1] = 0
    else:
        # fallback: use analytic phase to force non-degenerate segmentation
        try:
            detr = arr - pd.Series(arr).rolling(window=max(3, filter_window), min_periods=1).mean().values
            analytic = hilbert(np.nan_to_num(detr))
            phase = np.angle(analytic)
            labels = _phase_to_discrete(phase)
            if verbose:
                print("[cycle] peak/trough detection failed — used phase->discrete fallback.")
        except Exception as e:
            # final fallback: simple slope-quantile segmentation (guarantees some variation)
            q_low = np.nanpercentile(slope, 33)
            q_high = np.nanpercentile(slope, 66)
            labels = np.where(slope <= q_low, 3, np.where(slope >= q_high, 1, 1)).astype(int)
            # attempt to mark local extrema conservatively
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
    # where original close was NaN, keep as 0 (bottom) to be consistent
    out[nan_mask] = 0

    # final safety: if degenerate (single label only), force phase-based mapping to ensure diversity
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
            # leave as-is but ensure dtype and index correctness
            pass

    # ensure index type and length
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    try:
        assert len(out) == len(df_in)
        assert isinstance(out.index, pd.DatetimeIndex)
    except AssertionError:
        out = out.reindex(df_in.index, fill_value=0)

    if verbose:
        vals = dict(pd.Series(out.values).value_counts().to_dict())
        print(f"[cycle][discrete] final counts: {vals}")

    return out
