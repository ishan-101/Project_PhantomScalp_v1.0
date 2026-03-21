# Module Summary: `app/features/time_cycle.py`

## Purpose
The `time_cycle.py` module is designed to generate a comprehensive set of **technical analysis indicators** and **time-based features** from standard OHLCV (Open, High, Low, Close, Volume) market data. It serves as a core feature engineering component for strategies that rely on price action, trend following, momentum, or volatility analysis.

## Inputs
The primary input is a pandas DataFrame containing market data.
*   **Required Columns:** `close`, `high`, `low`, `volume`.
*   **Index:** Must be a `DatetimeIndex` (or contain a `datetime` column that can be converted).
*   **Optional Parameters:** A dictionary `params` can be passed to customize lookback periods or multipliers, though defaults are provided for all functions.

## Outputs
The module produces a DataFrame where each column represents a specific feature. All output columns are prefixed with `tc_` to indicate "Time Cycle" features.

**Key Output Groups:**
*   **Moving Averages:** `tc_ema_{L}`, `tc_sma_{L}`, `tc_hma_{L}`, `tc_wma_{L}`, `tc_dema_{L}`, `tc_tema_{L}` (Exponential, Simple, Hull, Weighted, Double, Triple MAs).
*   **Momentum/Oscillators:** `tc_rsi_{L}` (RSI), `tc_macd`, `tc_macd_sig`, `tc_macd_hist` (MACD), `tc_sto_k`, `tc_sto_d` (Stochastic), `tc_willr` (Williams %R), `tc_roc` (Rate of Change), `tc_mom` (Momentum).
*   **Volatility/Bands:** `tc_atr` (ATR), `tc_bb_up/mid/low` (Bollinger Bands), `tc_kc_up/low` (Keltner Channels), `tc_don_high/low` (Donchian Channels), `tc_chop` (Choppiness Index).
*   **Trend:** `tc_adx`, `tc_di_plus`, `tc_di_minus` (ADX system), `tc_supertrend_dir` (Supertrend direction), `tc_trend_strength` (Custom Trend Strength Index).
*   **Volume/Liquidity:** `tc_vwap` (VWAP), `tc_vol_comp_ratio` (Volatility Compression Ratio).
*   **Time & Session:** `tc_hour`, `tc_minute`, `tc_session` (Session labels: Asia, EU, US), `tc_session_vol_score` (Session-relative volatility), `tc_session_participation` (Volume participation).

## Typical Use-Case
This module is typically used in the **feature engineering stage** of a trading pipeline.
1.  **Data Loading:** Raw OHLCV bars are loaded.
2.  **Feature Generation:** `compute_time_cycle_features` is called to enrich the raw data with technical indicators.
3.  **Modeling/Signal Generation:** The resulting DataFrame (now containing `tc_rsi`, `tc_macd`, etc.) is fed into a machine learning model or a rule-based strategy to generate buy/sell signals.

## How to Run Locally

```python
import pandas as pd
import numpy as np
from app.features.time_cycle import compute_time_cycle_features

# 1. Create dummy OHLCV data
dates = pd.date_range("2023-01-01", periods=100, freq="1h")
df = pd.DataFrame({
    "open": np.random.rand(100) * 10 + 100,
    "high": np.random.rand(100) * 10 + 105,
    "low": np.random.rand(100) * 10 + 95,
    "close": np.random.rand(100) * 10 + 100,
    "volume": np.random.randint(100, 1000, 100)
}, index=dates)

# 2. Compute features
features_df = compute_time_cycle_features(df)

# 3. Inspect results
print(features_df.head())
print("Generated columns:", features_df.columns.tolist())
```
