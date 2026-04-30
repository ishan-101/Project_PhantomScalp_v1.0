"""Deterministic computation of Technical Indicator base features."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.rolling import rolling_max, rolling_mean, rolling_min, rolling_std


class FeatureComputationError(RuntimeError):
    """Raised when technical indicator computation fails."""


FEATURE_NAMES: List[str] = [
    "technical__ema_fast",
    "technical__ema_medium",
    "technical__ema_slow",
    "technical__ema_ratio_fast_slow",
    "technical__ema_slope_fast",
    "technical__hma",
    "technical__kama",
    "technical__jma",
    "technical__lsma",
    "technical__atr_short",
    "technical__atr_long",
    "technical__true_range",
    "technical__normalized_volatility",
    "technical__garman_klass_volatility",
    "technical__bb_width",
    "technical__bollinger_upper_dev",
    "technical__bollinger_lower_dev",
    "technical__donchian_width",
    "technical__range_delta_velocity",
    "technical__rsi",
    "technical__stoch_k",
    "technical__stoch_d",
    "technical__roc",
    "technical__tsi",
    "technical__macd_line",
    "technical__macd_signal",
    "technical__macd_histogram",
    "technical__williams_r",
    "technical__cci",
    "technical__adx",
    "technical__dmi_plus",
    "technical__dmi_minus",
    "technical__vwap_deviation_pct",
    "technical__ema_fast_deviation_pct",
    "technical__ema_slow_deviation_pct",
    "technical__range_normalized_position",
    "technical__rolling_skew",
    "technical__rolling_kurtosis",
    "technical__zscore_returns",
    "technical__direction_entropy",
    "technical__autocorrelation_returns",
    "technical__vfi",
    "technical__chaikin_oscillator",
    "technical__poly_regression_slope",
    "technical__poly_regression_residual",
    "technical__choppiness_index",
]

WINDOW_REQUIREMENTS: Dict[str, int] = {
    "technical__ema_fast": 12,
    "technical__ema_medium": 26,
    "technical__ema_slow": 50,
    "technical__ema_ratio_fast_slow": 50,
    "technical__ema_slope_fast": 12,
    "technical__hma": 20,
    "technical__kama": 30,
    "technical__jma": 30,
    "technical__lsma": 25,
    "technical__atr_short": 14,
    "technical__atr_long": 28,
    "technical__true_range": 2,
    "technical__normalized_volatility": 30,
    "technical__garman_klass_volatility": 30,
    "technical__bb_width": 20,
    "technical__bollinger_upper_dev": 20,
    "technical__bollinger_lower_dev": 20,
    "technical__donchian_width": 20,
    "technical__range_delta_velocity": 5,
    "technical__rsi": 14,
    "technical__stoch_k": 14,
    "technical__stoch_d": 16,
    "technical__roc": 12,
    "technical__tsi": 50,
    "technical__macd_line": 26,
    "technical__macd_signal": 35,
    "technical__macd_histogram": 35,
    "technical__williams_r": 14,
    "technical__cci": 20,
    "technical__adx": 28,
    "technical__dmi_plus": 28,
    "technical__dmi_minus": 28,
    "technical__vwap_deviation_pct": 1,
    "technical__ema_fast_deviation_pct": 12,
    "technical__ema_slow_deviation_pct": 50,
    "technical__range_normalized_position": 20,
    "technical__rolling_skew": 30,
    "technical__rolling_kurtosis": 30,
    "technical__zscore_returns": 30,
    "technical__direction_entropy": 20,
    "technical__autocorrelation_returns": 20,
    "technical__vfi": 130,
    "technical__chaikin_oscillator": 10,
    "technical__poly_regression_slope": 20,
    "technical__poly_regression_residual": 20,
    "technical__choppiness_index": 14,
}

EMA_FAST_SPAN = 12
EMA_MEDIUM_SPAN = 26
EMA_SLOW_SPAN = 50
EMA_SIGNAL_SPAN = 9
KAMA_ER_PERIOD = 10
KAMA_FAST = 2
KAMA_SLOW = 30
JMA_LENGTH = 20
JMA_PHASE = 0.0
HMA_PERIOD = 20
LSMA_PERIOD = 25
TSI_LONG = 25
TSI_SHORT = 13
ROC_PERIOD = 12
RSI_PERIOD = 14
ADX_PERIOD = 14
STOCH_D_SMOOTH = 3
STOCH_PERIOD = 14
WILLIAMS_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
DONCHIAN_PERIOD = 20
RANGE_POSITION_PERIOD = 20
SKEW_WINDOW = 30
KURTOSIS_WINDOW = 30
Z_RETURN_WINDOW = 30
ENTROPY_WINDOW = 20
AUTO_WINDOW = 20
GARMAN_KLASS_WINDOW = 30
VFI_WINDOW = 130
VFI_VFACTOR = 0.2
CHAIKIN_SHORT = 3
CHAIKIN_LONG = 10
POLY_WINDOW = 20
CHOP_WINDOW = 14


def _select_close(df: pd.DataFrame) -> pd.Series:
    if "price__mid" in df.columns:
        close = df["price__mid"]
    elif "ohlcv__close" in df.columns:
        close = df["ohlcv__close"]
    else:
        raise FeatureComputationError("A mid or close price column is required for technical indicators.")

    if close.isna().any():
        raise FeatureComputationError("Price inputs contain nulls; fill explicitly before computation.")
    return close.astype(np.float32)


def _validate_inputs(df: pd.DataFrame) -> None:
    required_cols = ["ohlcv__open", "ohlcv__high", "ohlcv__low", "ohlcv__close", "ohlcv__volume", "timestamp"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required OHLCV columns: {missing}")
    for col in required_cols:
        if df[col].isna().any():
            raise FeatureComputationError(f"Column '{col}' contains nulls; fill explicitly before computing features.")
    if not pd.Series(df["timestamp"]).is_monotonic_increasing:
        raise FeatureComputationError("Timestamps must be monotonic for causal rolling windows.")

    max_window = max(WINDOW_REQUIREMENTS.values())
    if len(df) < max_window:
        raise FeatureComputationError(
            f"Insufficient history for technical indicators: need at least {max_window} rows, found {len(df)}."
        )


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    tr = ranges.max(axis=1)
    tr = tr.fillna(0.0)
    return tr.astype(np.float32)


def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _hma(series: pd.Series, period: int) -> pd.Series:
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    def _wma(values: pd.Series, length: int) -> pd.Series:
        weights = np.arange(1, length + 1)
        return values.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    wma_full = _wma(series, period)
    wma_half = _wma(series, half_length)
    intermediate = 2 * wma_half - wma_full
    hma_series = _wma(intermediate, sqrt_length)
    return hma_series.fillna(0.0).astype(np.float32)


def _kama(series: pd.Series) -> pd.Series:
    price = series.to_numpy(dtype=np.float64)
    out = np.zeros_like(price)
    fast_sc = 2 / (KAMA_FAST + 1)
    slow_sc = 2 / (KAMA_SLOW + 1)

    for idx, value in enumerate(price):
        if idx == 0:
            out[idx] = value
            continue
        start = max(0, idx - KAMA_ER_PERIOD + 1)
        change = abs(price[idx] - price[start])
        volatility = np.sum(np.abs(np.diff(price[start: idx + 1])))
        efficiency_ratio = 0.0 if volatility == 0 else change / volatility
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        out[idx] = out[idx - 1] + smoothing_constant * (value - out[idx - 1])

    return pd.Series(out, index=series.index, dtype=np.float32)


def _jma(series: pd.Series) -> pd.Series:
    # Simplified Jurik-like moving average using double exponential smoothing with phase offset.
    alpha = 2.0 / (JMA_LENGTH + 1)
    ema1 = series.ewm(alpha=alpha, adjust=False).mean()
    ema2 = ema1.ewm(alpha=alpha, adjust=False).mean()
    jma_series = ema2 + JMA_PHASE * (ema1 - ema2)
    return jma_series.fillna(0.0).astype(np.float32)


def _lsma(series: pd.Series, window: int) -> pd.Series:
    def _fit(window_values: np.ndarray) -> float:
        x = np.arange(window)
        coeffs = np.polyfit(x, window_values, deg=1)
        slope, intercept = coeffs
        return float(slope * (window - 1) + intercept)

    lsma_series = series.rolling(window, min_periods=window).apply(_fit, raw=True)
    return lsma_series.fillna(0.0).astype(np.float32)


def _rsi(series: pd.Series) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_smoothing(gain, RSI_PERIOD)
    avg_loss = _wilder_smoothing(loss, RSI_PERIOD)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(0.0).clip(0.0, 100.0)
    return rsi.astype(np.float32)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    highest_high = rolling_max(high, STOCH_PERIOD)
    lowest_low = rolling_min(low, STOCH_PERIOD)
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    stoch_k = ((close - lowest_low) / denominator * 100.0).fillna(0.0)
    stoch_d = stoch_k.rolling(STOCH_D_SMOOTH, min_periods=STOCH_D_SMOOTH).mean().fillna(0.0)
    stoch_k = stoch_k.clip(0.0, 100.0).astype(np.float32)
    stoch_d = stoch_d.clip(0.0, 100.0).astype(np.float32)
    return stoch_k, stoch_d


def _roc(series: pd.Series, period: int) -> pd.Series:
    roc = (series / series.shift(period) - 1.0) * 100.0
    roc = roc.fillna(0.0)
    return roc.astype(np.float32)


def _tsi(series: pd.Series) -> pd.Series:
    momentum = series.diff()
    abs_momentum = momentum.abs()
    ema1 = _ema(momentum, TSI_SHORT)
    ema2 = _ema(ema1, TSI_LONG)
    abs_ema1 = _ema(abs_momentum, TSI_SHORT)
    abs_ema2 = _ema(abs_ema1, TSI_LONG)
    tsi = 100.0 * ema2 / abs_ema2.replace(0.0, np.nan)
    tsi = tsi.fillna(0.0).clip(-100.0, 100.0)
    return tsi.astype(np.float32)


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(series, EMA_FAST_SPAN)
    slow = _ema(series, EMA_SLOW_SPAN)
    macd_line = fast - slow
    signal = _ema(macd_line, EMA_SIGNAL_SPAN)
    histogram = macd_line - signal
    return macd_line.astype(np.float32), signal.astype(np.float32), histogram.astype(np.float32)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    highest_high = rolling_max(high, WILLIAMS_PERIOD)
    lowest_low = rolling_min(low, WILLIAMS_PERIOD)
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    williams = -100.0 * (highest_high - close) / denominator
    williams = williams.fillna(0.0).clip(-100.0, 0.0)
    return williams.astype(np.float32)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    sma_typical = rolling_mean(typical_price, DONCHIAN_PERIOD)
    mean_dev = typical_price.rolling(DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (typical_price - sma_typical) / (0.015 * mean_dev.replace(0.0, np.nan))
    cci = cci.fillna(0.0).clip(-500.0, 500.0)
    return cci.astype(np.float32)


def _dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), other=0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), other=0.0)

    tr = _true_range(high, low, close)
    atr = _wilder_smoothing(tr, ADX_PERIOD)

    plus_di = 100.0 * _wilder_smoothing(plus_dm, ADX_PERIOD) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_smoothing(minus_dm, ADX_PERIOD) / atr.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder_smoothing(dx, ADX_PERIOD)

    plus_di = plus_di.fillna(0.0).clip(0.0, 100.0).astype(np.float32)
    minus_di = minus_di.fillna(0.0).clip(0.0, 100.0).astype(np.float32)
    adx = adx.fillna(0.0).clip(0.0, 100.0).astype(np.float32)
    return plus_di, minus_di, adx


def _normalized_volatility(close: pd.Series) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(Z_RETURN_WINDOW, min_periods=Z_RETURN_WINDOW).std()
    vol = vol.fillna(0.0).clip(lower=0.0)
    return vol.astype(np.float32)


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    high = df["ohlcv__high"].astype(np.float32)
    low = df["ohlcv__low"].astype(np.float32)
    close = df["ohlcv__close"].astype(np.float32)
    open_price = df["ohlcv__open"].astype(np.float32)
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)
    sigma_sq = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk = sigma_sq.rolling(GARMAN_KLASS_WINDOW, min_periods=GARMAN_KLASS_WINDOW).mean()
    gk = np.sqrt(gk.clip(lower=0.0)).fillna(0.0)
    return gk.astype(np.float32)


def _bollinger_metrics(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    mean = rolling_mean(close, BB_PERIOD)
    std = rolling_std(close, BB_PERIOD)
    upper = mean + BB_STD * std
    lower = mean - BB_STD * std
    width = (upper - lower) / mean.replace(0.0, np.nan)
    upper_dev = (close - upper) / std.replace(0.0, np.nan)
    lower_dev = (close - lower) / std.replace(0.0, np.nan)
    width = width.fillna(0.0).clip(lower=0.0)
    upper_dev = upper_dev.fillna(0.0).clip(-10.0, 10.0)
    lower_dev = lower_dev.fillna(0.0).clip(-10.0, 10.0)
    return width.astype(np.float32), upper_dev.astype(np.float32), lower_dev.astype(np.float32)


def _donchian_width(high: pd.Series, low: pd.Series) -> pd.Series:
    highest_high = rolling_max(high, DONCHIAN_PERIOD)
    lowest_low = rolling_min(low, DONCHIAN_PERIOD)
    denominator = lowest_low.replace(0.0, np.nan)
    width = (highest_high - lowest_low) / denominator
    width = width.fillna(0.0).clip(lower=0.0)
    return width.astype(np.float32)


def _range_delta_velocity(high: pd.Series, low: pd.Series) -> pd.Series:
    rng = high - low
    velocity = rng.diff() / rng.shift(1).replace(0.0, np.nan)
    velocity = velocity.fillna(0.0)
    return velocity.astype(np.float32)


def _vwap_deviation(close: pd.Series, volume: pd.Series) -> pd.Series:
    cum_vol = volume.cumsum()
    cum_px_vol = (close * volume).cumsum()
    vwap = cum_px_vol / cum_vol.replace(0.0, np.nan)
    deviation = (close - vwap) / vwap
    deviation = deviation.fillna(0.0) * 100.0
    return deviation.astype(np.float32)


def _range_normalized_position(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    rolling_high = rolling_max(high, RANGE_POSITION_PERIOD)
    rolling_low = rolling_min(low, RANGE_POSITION_PERIOD)
    denominator = (rolling_high - rolling_low).replace(0.0, np.nan)
    position = (close - rolling_low) / denominator
    position = position.fillna(0.0).clip(0.0, 1.0)
    return position.astype(np.float32)


def _rolling_shape_metrics(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    log_ret = np.log(close / close.shift(1))
    skew = log_ret.rolling(SKEW_WINDOW, min_periods=SKEW_WINDOW).skew()
    kurt = log_ret.rolling(KURTOSIS_WINDOW, min_periods=KURTOSIS_WINDOW).kurt()
    mean = log_ret.rolling(Z_RETURN_WINDOW, min_periods=Z_RETURN_WINDOW).mean()
    std = log_ret.rolling(Z_RETURN_WINDOW, min_periods=Z_RETURN_WINDOW).std()
    zscore = (log_ret - mean) / std.replace(0.0, np.nan)

    skew = skew.fillna(0.0).clip(-10.0, 10.0)
    kurt = kurt.fillna(0.0).clip(-10.0, 30.0)
    zscore = zscore.fillna(0.0).clip(-10.0, 10.0)
    return skew.astype(np.float32), kurt.astype(np.float32), zscore.astype(np.float32)


def _direction_entropy(close: pd.Series) -> pd.Series:
    log_ret = np.log(close / close.shift(1))

    def _entropy(window: np.ndarray) -> float:
        if window.size == 0:
            return 0.0
        signs, counts = np.unique(np.sign(window), return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        normalized = entropy / np.log2(3)
        return float(np.clip(normalized, 0.0, 1.0))

    entropy_series = log_ret.rolling(ENTROPY_WINDOW, min_periods=ENTROPY_WINDOW).apply(
        _entropy, raw=True
    )
    return entropy_series.fillna(0.0).astype(np.float32)


def _autocorrelation_returns(close: pd.Series) -> pd.Series:
    log_ret = np.log(close / close.shift(1))

    def _autocorr(window: np.ndarray) -> float:
        if window.size < 2:
            return 0.0
        series = window
        x = series[:-1]
        y = series[1:]
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std == 0 or y_std == 0:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(np.clip(corr, -1.0, 1.0))

    ac_series = log_ret.rolling(AUTO_WINDOW, min_periods=AUTO_WINDOW).apply(
        _autocorr, raw=True
    )
    return ac_series.fillna(0.0).astype(np.float32)


def _vfi(df: pd.DataFrame, close: pd.Series) -> pd.Series:
    high = df["ohlcv__high"].astype(np.float32)
    low = df["ohlcv__low"].astype(np.float32)
    volume = df["ohlcv__volume"].astype(np.float32)
    typical_price = (high + low + close) / 3.0
    tp_change = typical_price.diff()
    tp_std = tp_change.rolling(30, min_periods=30).std()
    cutoff = VFI_VFACTOR * tp_std
    volume_mean = volume.rolling(VFI_WINDOW, min_periods=VFI_WINDOW).mean()
    price_condition = np.where(tp_change > cutoff, volume, 0.0) - np.where(tp_change < -cutoff, volume, 0.0)
    vfi_raw = pd.Series(price_condition, index=df.index)
    vfi = vfi_raw.ewm(span=VFI_WINDOW, adjust=False).mean() / volume_mean.replace(0.0, np.nan)
    vfi = vfi.fillna(0.0).clip(-200.0, 200.0)
    return vfi.astype(np.float32)


def _chaikin_oscillator(df: pd.DataFrame) -> pd.Series:
    high = df["ohlcv__high"].astype(np.float32)
    low = df["ohlcv__low"].astype(np.float32)
    close = df["ohlcv__close"].astype(np.float32)
    volume = df["ohlcv__volume"].astype(np.float32)
    denominator = (high - low).replace(0.0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / denominator
    money_flow_volume = money_flow_multiplier.fillna(0.0) * volume
    adl = money_flow_volume.cumsum()
    ema_short = _ema(adl, CHAIKIN_SHORT)
    ema_long = _ema(adl, CHAIKIN_LONG)
    oscillator = ema_short - ema_long
    return oscillator.fillna(0.0).astype(np.float32)


def _poly_regression(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    def _fit(window_values: np.ndarray) -> tuple[float, float]:
        x = np.arange(POLY_WINDOW)
        slope, intercept = np.polyfit(x, window_values, deg=1)
        predicted = slope * (POLY_WINDOW - 1) + intercept
        residual = window_values[-1] - predicted
        return slope, residual

    slope_series = []
    residual_series = []
    for idx in range(len(close)):
        if idx + 1 < POLY_WINDOW:
            slope_series.append(0.0)
            residual_series.append(0.0)
            continue
        window_vals = close.iloc[idx - POLY_WINDOW + 1: idx + 1].to_numpy()
        slope, residual = _fit(window_vals)
        slope_series.append(float(slope))
        residual_series.append(float(residual))

    slope_pd = pd.Series(slope_series, index=close.index)
    residual_pd = pd.Series(residual_series, index=close.index)
    return slope_pd.astype(np.float32), residual_pd.astype(np.float32)


def _choppiness_index(high: pd.Series, low: pd.Series, tr: pd.Series) -> pd.Series:
    sum_tr = tr.rolling(CHOP_WINDOW, min_periods=CHOP_WINDOW).sum()
    range_high = rolling_max(high, CHOP_WINDOW)
    range_low = rolling_min(low, CHOP_WINDOW)
    denom = (range_high - range_low).replace(0.0, np.nan)
    chop = 100.0 * np.log10(sum_tr / denom) / np.log10(CHOP_WINDOW)
    chop = chop.fillna(0.0).clip(0.0, 100.0)
    return chop.astype(np.float32)


def compute_technical_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic technical indicators with fixed lookbacks and no future access."""

    _validate_inputs(df)

    close = _select_close(df)
    high = df["ohlcv__high"].astype(np.float32)
    low = df["ohlcv__low"].astype(np.float32)
    open_price = df["ohlcv__open"].astype(np.float32)
    volume = df["ohlcv__volume"].astype(np.float32)

    output = df.copy()

    true_range = _true_range(high, low, close)
    atr_short = _wilder_smoothing(true_range, ADX_PERIOD)
    atr_long = _wilder_smoothing(true_range, 2 * ADX_PERIOD)

    ema_fast = _ema(close, EMA_FAST_SPAN)
    ema_medium = _ema(close, EMA_MEDIUM_SPAN)
    ema_slow = _ema(close, EMA_SLOW_SPAN)

    ema_ratio = (ema_fast / ema_slow.replace(0.0, np.nan)).fillna(0.0)
    ema_slope = ema_fast.diff().fillna(0.0)

    hma = _hma(close, HMA_PERIOD)
    kama = _kama(close)
    jma = _jma(close)
    lsma = _lsma(close, LSMA_PERIOD)

    normalized_vol = _normalized_volatility(close)
    gk_vol = _garman_klass(df)
    bb_width, bb_upper_dev, bb_lower_dev = _bollinger_metrics(close)
    donchian_width = _donchian_width(high, low)
    range_velocity = _range_delta_velocity(high, low)

    rsi = _rsi(close)
    stoch_k, stoch_d = _stochastic(high, low, close)
    roc = _roc(close, ROC_PERIOD)
    tsi = _tsi(close)
    macd_line, macd_signal, macd_hist = _macd(close)
    williams_r = _williams_r(high, low, close)
    cci = _cci(high, low, close)
    dmi_plus, dmi_minus, adx = _dmi_adx(high, low, close)

    vwap_dev = _vwap_deviation(close, volume)
    ema_fast_dev = (close - ema_fast) / ema_fast.replace(0.0, np.nan) * 100.0
    ema_fast_dev = ema_fast_dev.fillna(0.0).astype(np.float32)
    ema_slow_dev = (close - ema_slow) / ema_slow.replace(0.0, np.nan) * 100.0
    ema_slow_dev = ema_slow_dev.fillna(0.0).astype(np.float32)

    range_position = _range_normalized_position(high, low, close)
    rolling_skew, rolling_kurtosis, zscore_returns = _rolling_shape_metrics(close)
    direction_entropy = _direction_entropy(close)
    autocorr_returns = _autocorrelation_returns(close)

    vfi = _vfi(df, close)
    chaikin = _chaikin_oscillator(df)

    poly_slope, poly_residual = _poly_regression(close)
    chop_index = _choppiness_index(high, low, true_range)

    output["technical__ema_fast"] = ema_fast.fillna(0.0).astype(np.float32)
    output["technical__ema_medium"] = ema_medium.fillna(0.0).astype(np.float32)
    output["technical__ema_slow"] = ema_slow.fillna(0.0).astype(np.float32)
    output["technical__ema_ratio_fast_slow"] = ema_ratio.astype(np.float32)
    output["technical__ema_slope_fast"] = ema_slope.astype(np.float32)
    output["technical__hma"] = hma
    output["technical__kama"] = kama
    output["technical__jma"] = jma
    output["technical__lsma"] = lsma
    output["technical__atr_short"] = atr_short.fillna(0.0).astype(np.float32)
    output["technical__atr_long"] = atr_long.fillna(0.0).astype(np.float32)
    output["technical__true_range"] = true_range
    output["technical__normalized_volatility"] = normalized_vol
    output["technical__garman_klass_volatility"] = gk_vol
    output["technical__bb_width"] = bb_width
    output["technical__bollinger_upper_dev"] = bb_upper_dev
    output["technical__bollinger_lower_dev"] = bb_lower_dev
    output["technical__donchian_width"] = donchian_width
    output["technical__range_delta_velocity"] = range_velocity
    output["technical__rsi"] = rsi
    output["technical__stoch_k"] = stoch_k
    output["technical__stoch_d"] = stoch_d
    output["technical__roc"] = roc
    output["technical__tsi"] = tsi
    output["technical__macd_line"] = macd_line
    output["technical__macd_signal"] = macd_signal
    output["technical__macd_histogram"] = macd_hist
    output["technical__williams_r"] = williams_r
    output["technical__cci"] = cci
    output["technical__adx"] = adx
    output["technical__dmi_plus"] = dmi_plus
    output["technical__dmi_minus"] = dmi_minus
    output["technical__vwap_deviation_pct"] = vwap_dev
    output["technical__ema_fast_deviation_pct"] = ema_fast_dev
    output["technical__ema_slow_deviation_pct"] = ema_slow_dev
    output["technical__range_normalized_position"] = range_position
    output["technical__rolling_skew"] = rolling_skew
    output["technical__rolling_kurtosis"] = rolling_kurtosis
    output["technical__zscore_returns"] = zscore_returns
    output["technical__direction_entropy"] = direction_entropy
    output["technical__autocorrelation_returns"] = autocorr_returns
    output["technical__vfi"] = vfi
    output["technical__chaikin_oscillator"] = chaikin
    output["technical__poly_regression_slope"] = poly_slope
    output["technical__poly_regression_residual"] = poly_residual
    output["technical__choppiness_index"] = chop_index

    missing = [col for col in FEATURE_NAMES if col not in output.columns]
    if missing:
        raise FeatureComputationError(f"Missing expected feature outputs: {missing}")

    for col in FEATURE_NAMES:
        if output[col].isna().any():
            raise FeatureComputationError(f"Nulls detected after computing {col}; ensure explicit filling.")

    return output


__all__ = ["compute_technical_indicator_features", "FeatureComputationError", "FEATURE_NAMES", "WINDOW_REQUIREMENTS"]
