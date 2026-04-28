# Module Summary: `app/features/options_features.py`

## Purpose
The `options_features.py` module is a specialized toolkit for **options market analysis**. It computes essential risk metrics (Greeks), volatility surfaces, and flow dynamics. This module is critical for strategies that trade options directly or use options market signals (like Gamma Exposure or IV Skew) to predict underlying asset movements.

## Inputs
The primary input is a pandas DataFrame containing options contract data.
*   **Required Columns:** `strike`, `implied_vol`, `expiry` (timestamp), `option_type` ('call'/'put'), `open_interest`.
*   **Pricing Columns:** `mid_price` or `last_price`.
*   **Underlying Data:** Optional `spot_series` (time-aligned spot prices) or columns like `spot`/`underlying` in the main DataFrame.
*   **Optional Parameters:** Risk-free rate `r`, `gex_base_notional` (for normalization), `opt_trades_df` (for flow analysis).

## Outputs
The module produces a DataFrame with columns prefixed with `opt_` (Options).

**Key Output Groups:**
*   **Greeks:** `opt_delta`, `opt_gamma`, `opt_vega`, `opt_theta` (Standard Black-Scholes Greeks).
*   **Gamma Exposure (GEX):** `opt_gex` (Per-instrument GEX), `opt_gex_total` (Aggregate GEX), `opt_gex_norm` (Normalized GEX).
*   **Implied Volatility (IV):** `opt_iv_rank_{w}` (IV Rank over window w), `opt_iv_pct_{w}` (IV Percentile), `opt_iv_term_slope` (Term structure slope), `opt_iv_crush_prob`, `opt_iv_expand_prob` (Proxies for volatility mean reversion).
*   **Skew/Premium:** `opt_skew_index` (Call vs. Put IV difference), `opt_prem_pct` (Premium as % of spot), `opt_prem_z` (Z-score of premium).
*   **Open Interest (OI):** `opt_oi_change`, `opt_oi_per_sec` (Rate of OI change), `opt_unusual_oi_flag` (Spikes in OI).
*   **Flow:** `opt_flow_delta` (Net volume), `opt_vol_delta` (Signed volume delta).

## Typical Use-Case
This module is used for **volatility trading** and **risk management**.
1.  **Data Loading:** Options chain data is loaded (often snapshots or end-of-day).
2.  **Metric Calculation:** `compute_options_features` calculates Greeks and GEX.
3.  **Analysis:**
    *   **Market Makers:** Monitor `opt_gex_total` to anticipate hedging flows (gamma squeezes).
    *   **Vol Traders:** Use `opt_iv_rank` and `opt_skew_index` to find cheap/expensive options.
    *   **Directional Traders:** Use `opt_flow_delta` to track institutional positioning.

## How to Run Locally

```python
import pandas as pd
import numpy as np
from app.features.options_features import compute_options_features

# 1. Create dummy Options data
dates = pd.date_range("2023-01-01", periods=5, freq="1D")
opt_df = pd.DataFrame({
    "strike": [100, 100, 100, 100, 100],
    "implied_vol": [0.2, 0.21, 0.19, 0.22, 0.20],
    "expiry": pd.date_range("2023-02-01", periods=5, freq="1D"),
    "option_type": ["call", "call", "call", "call", "call"],
    "open_interest": [1000, 1100, 1050, 1200, 1150],
    "mid_price": [2.5, 2.6, 2.4, 2.7, 2.5]
}, index=dates)

# 2. Create dummy Spot data
spot_series = pd.Series([100, 101, 99, 102, 100], index=dates, name="close")

# 3. Compute features
opt_features = compute_options_features(opt_df, spot_series=spot_series)

# 4. Inspect results
print(opt_features.head())
print("Generated columns:", opt_features.columns.tolist())
```
