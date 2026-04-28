"""Deterministic computation of Cycle / FFT / Ehlers base features."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureComputationError(RuntimeError):
    """Raised when feature computation cannot proceed due to invalid inputs."""


HILBERT_WINDOW = 64
FFT_WINDOW = 128
TREND_WINDOW = 32
CONSISTENCY_WINDOW = 32

EHLERS_PERIOD = 10.0
EHLERS_A1 = float(np.exp(-1.414 * np.pi / EHLERS_PERIOD))
EHLERS_C2 = 2 * EHLERS_A1 * np.cos(1.414 * np.pi / EHLERS_PERIOD)
EHLERS_C3 = -EHLERS_A1 ** 2
EHLERS_C1 = 1 - EHLERS_C2 - EHLERS_C3


FEATURE_NAMES = [
    "cycle__hilbert_phase",
    "cycle__dominant_period",
    "cycle__ehlers_filt_output",
    "cycle__instantaneous_frequency",
    "cycle__trend_component",
    "cycle__cycle_component",
    "cycle__phase_acceleration",
    "cycle__phase_consistency",
]


def _select_price_column(df: pd.DataFrame) -> pd.Series:
    if "price__mid" in df.columns:
        price = df["price__mid"]
    elif "ohlcv__close" in df.columns:
        price = df["ohlcv__close"]
    else:
        raise FeatureComputationError("A mid or close price column is required for cycle features.")

    if price.isna().any():
        raise FeatureComputationError("Price inputs contain nulls; fill explicitly before computation.")
    return price.astype(np.float32)


def _analytic_signal(window: np.ndarray) -> np.ndarray:
    n = len(window)
    spectrum = np.fft.fft(window)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    return np.fft.ifft(spectrum * h)


def _hilbert_phase(series: pd.Series) -> pd.Series:
    def _phase(window: np.ndarray) -> float:
        analytic = _analytic_signal(window)
        return float(np.angle(analytic[-1]))

    phase = series.rolling(HILBERT_WINDOW, min_periods=HILBERT_WINDOW).apply(
        _phase, raw=True
    )
    phase = phase.fillna(0.0)
    phase = phase.clip(-np.pi, np.pi)
    return phase.astype(np.float32)


def _dominant_period(series: pd.Series) -> pd.Series:
    def _period(window: np.ndarray) -> float:
        window = window - window.mean()
        spectrum = np.fft.rfft(window)
        magnitudes = np.abs(spectrum)
        if magnitudes.shape[0] <= 1:
            return float(FFT_WINDOW)
        magnitudes[0] = 0.0
        peak_index = int(np.argmax(magnitudes))
        if peak_index == 0:
            peak_index = 1
        period = FFT_WINDOW / peak_index
        return float(period)

    periods = series.rolling(FFT_WINDOW, min_periods=FFT_WINDOW).apply(_period, raw=True)
    periods = periods.fillna(1.0)
    periods = periods.clip(1.0, float(FFT_WINDOW))
    return periods.astype(np.float32)


def _ehlers_super_smoother(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=np.float64)
    output = np.zeros_like(values)
    if len(values) == 0:
        return pd.Series(output, index=series.index, dtype=np.float32)

    output[0] = values[0]
    if len(values) > 1:
        output[1] = np.mean(values[:2])

    for idx in range(2, len(values)):
        output[idx] = (
            EHLERS_C1 * 0.5 * (values[idx] + values[idx - 1])
            + EHLERS_C2 * output[idx - 1]
            + EHLERS_C3 * output[idx - 2]
        )

    return pd.Series(output, index=series.index, dtype=np.float32)


def _instantaneous_frequency(phase: pd.Series) -> pd.Series:
    unwrapped = np.unwrap(phase.to_numpy(dtype=np.float64))
    inst_freq = pd.Series(unwrapped, index=phase.index).diff()
    inst_freq = inst_freq.fillna(0.0)
    inst_freq = inst_freq.clip(-np.pi, np.pi)
    return inst_freq.astype(np.float32)


def _phase_acceleration(inst_freq: pd.Series) -> pd.Series:
    accel = inst_freq.diff()
    accel = accel.fillna(0.0)
    accel = accel.clip(-2 * np.pi, 2 * np.pi)
    return accel.astype(np.float32)


def _trend_component(series: pd.Series) -> pd.Series:
    trend = series.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    trend = trend.fillna(0.0)
    return trend.astype(np.float32)


def _cycle_component(price: pd.Series, trend: pd.Series) -> pd.Series:
    cycle = price - trend
    return cycle.astype(np.float32)


def _phase_consistency(inst_freq: pd.Series) -> pd.Series:
    def _consistency(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        deviation = float(np.std(window))
        normalized = deviation / np.pi
        score = 1.0 - min(1.0, normalized)
        return float(np.clip(score, 0.0, 1.0))

    consistency = inst_freq.rolling(
        CONSISTENCY_WINDOW, min_periods=CONSISTENCY_WINDOW
    ).apply(_consistency, raw=True)
    consistency = consistency.fillna(0.0)
    consistency = consistency.clip(0.0, 1.0)
    return consistency.astype(np.float32)


def compute_cycle_fft_ehlers_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic cycle features using fixed windows and coefficients."""

    if "timestamp" not in df.columns:
        raise FeatureComputationError("timestamp column is required for ordering checks.")
    if not pd.Series(df["timestamp"]).is_monotonic_increasing:
        raise FeatureComputationError("Timestamp ordering is required for causal computation.")

    if len(df) < FFT_WINDOW:
        raise FeatureComputationError(
            f"Insufficient history for FFT-based dominant period: need >= {FFT_WINDOW} rows."
        )

    price = _select_price_column(df)
    output = df.copy()

    hilbert_phase = _hilbert_phase(price)
    dominant_period = _dominant_period(price)
    ehlers_output = _ehlers_super_smoother(price)
    inst_freq = _instantaneous_frequency(hilbert_phase)
    trend = _trend_component(price)
    cycle_component = _cycle_component(price, trend)
    phase_accel = _phase_acceleration(inst_freq)
    phase_consistency = _phase_consistency(inst_freq)

    output[FEATURE_NAMES[0]] = hilbert_phase
    output[FEATURE_NAMES[1]] = dominant_period
    output[FEATURE_NAMES[2]] = ehlers_output
    output[FEATURE_NAMES[3]] = inst_freq
    output[FEATURE_NAMES[4]] = trend
    output[FEATURE_NAMES[5]] = cycle_component
    output[FEATURE_NAMES[6]] = phase_accel
    output[FEATURE_NAMES[7]] = phase_consistency

    if output[FEATURE_NAMES].isna().any().any():
        raise FeatureComputationError("Nulls detected after cycle feature computation; fill logic must be explicit.")

    return output


__all__ = ["compute_cycle_fft_ehlers_features", "FeatureComputationError"]
