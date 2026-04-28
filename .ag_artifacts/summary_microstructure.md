# Module Summary: `app/features/microstructure.py`

## Purpose
The `microstructure.py` module focuses on extracting granular market signals from **high-frequency data sources** like individual trades and Level 2 (L2) order book snapshots. It aims to quantify order flow toxicity, liquidity imbalances, and market pressure, which are critical for short-term alpha generation and execution algorithms.

## Inputs
The module accepts three types of DataFrames, aligned by timestamp:
1.  **`bar_df` (Required):** Standard OHLCV bars (used as the reference index).
2.  **`trades_df` (Optional):** Individual trade data with columns `price`, `size`, `side` (buy/sell).
3.  **`depth_df` (Optional):** L2 Order book snapshots with columns like `bid_1_price`, `bid_1_size`, `ask_1_price`, `ask_1_size` (up to N levels).

## Outputs
The module produces a DataFrame with columns prefixed with `ms_` (MicroStructure).

**Key Output Groups:**
*   **Order Flow/Aggression:** `ms_aggression_idx` (Ratio of aggressive buying/selling), `ms_orderflow_vel` (Speed of signed volume), `ms_buy_sell_imb` (Imbalance between buy/sell volume).
*   **Order Book Imbalance:** `ms_imb_level_{i}` (Imbalance at specific depth levels), `ms_vol_imb_sum5` (Aggregate volume imbalance), `ms_book_pressure_delta` (Weighted pressure from order book), `ms_top_book_pressure` (Pressure at the best bid/ask).
*   **Liquidity/Market Quality:** `ms_spread`, `ms_liq_decay` (Rate of liquidity replenishment), `ms_pull_stack_ratio` (Ratio of order pulling vs. stacking).
*   **Anomalies/Flags:** `ms_sweep_flag` (Large aggressive sweeps), `ms_iceberg_flag` (Hidden order detection - placeholder), `ms_spoof_rate` (Spoofing activity proxy), `ms_absorption_flag` (Passive absorption of aggressive flow).
*   **Micro-Trends:** `ms_microtrend_vector` (Short-term directional momentum), `ms_micro_divergence` (Divergence between price and CVD).

## Typical Use-Case
This module is essential for **High-Frequency Trading (HFT)** or **Market Making** strategies.
1.  **Data Ingestion:** Tick-level trades and order book updates are captured.
2.  **Resampling & Feature Extraction:** Data is resampled to a target frequency (e.g., 1-second or 1-minute bars), and `compute_microstructure_features` is called.
3.  **Signal Detection:** Features like `ms_orderflow_vel` or `ms_sweep_flag` are used to detect immediate price moves or liquidity gaps.

## How to Run Locally

```python
import pandas as pd
import numpy as np
from app.features.microstructure import compute_microstructure_features

# 1. Create dummy Bar data
dates = pd.date_range("2023-01-01 09:30", periods=60, freq="1min")
bar_df = pd.DataFrame({
    "close": np.random.rand(60) * 10 + 100
}, index=dates)

# 2. Create dummy Trades data (more granular)
trade_dates = pd.date_range("2023-01-01 09:30", periods=600, freq="6s")
trades_df = pd.DataFrame({
    "price": np.random.rand(600) * 10 + 100,
    "size": np.random.randint(1, 100, 600),
    "side": np.random.choice(["buy", "sell"], 600)
}, index=trade_dates)

# 3. Create dummy Depth data
depth_df = pd.DataFrame(index=dates)
for i in range(1, 6):
    depth_df[f"bid_{i}_size"] = np.random.randint(100, 500, 60)
    depth_df[f"ask_{i}_size"] = np.random.randint(100, 500, 60)
    depth_df[f"bid_{i}_price"] = 100 - i*0.01
    depth_df[f"ask_{i}_price"] = 100 + i*0.01

# 4. Compute features
ms_features = compute_microstructure_features(
    bar_df, 
    trades_df=trades_df, 
    depth_df=depth_df
)

# 5. Inspect results
print(ms_features.head())
print("Generated columns:", ms_features.columns.tolist())
```
