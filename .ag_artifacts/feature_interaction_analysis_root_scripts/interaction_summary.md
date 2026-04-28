# Feature Interaction Summary

## 1. app.features.time_cycle

**Primary Function:** `compute_time_cycle_features`
**Key Outputs:** `tc_ema_*`, `tc_rsi_*`, `tc_atr`, `tc_adx`, `tc_trend_strength`, `tc_vol_comp_ratio`

### Consumers
*   **Production Pipeline:**
    *   `scripts/prepare_labels_dataset.py`: Imports and calls `compute_time_cycle_features` to generate datasets.
*   **Testing:**
    *   `scripts/test_labels.py`: Dynamic import and call. Validates specific columns (`tc_ema_21`, `tc_rsi_14`).
    *   `scripts/test_all_features.py`: Direct import and call. Validates `tc_ema_9` etc.
*   **Backtesting:**
    *   `app/orchestrator/backtest_v02.py`: Direct import and call. Uses `tc_atr` for stop-loss/take-profit calculations.
*   **Labeling (Indirect Consumer):**
    *   `app/ml/labels/regime.py`: Consumes `tc_atr`, `tc_adx`, `tc_trend_strength`, `tc_vol_comp_ratio` from the dataframe.
    *   `app/ml/labels/reversal.py`: Consumes `tc_rsi_14`.

## 2. app.features.microstructure

**Primary Function:** `compute_microstructure_features`
**Key Outputs:** `ms_aggression_idx`, `ms_orderflow_vel`, `ms_microtrend_vector`

### Consumers
*   **Production Pipeline:**
    *   `scripts/prepare_labels_dataset.py`: Imports and calls `compute_microstructure_features`.
*   **Testing:**
    *   `scripts/test_labels.py`: Dynamic import and call. Validates `ms_aggression_idx`.
    *   `scripts/test_all_features.py`: Direct import and call. Validates `ms_orderflow_vel`.
*   **Backtesting:**
    *   `app/orchestrator/backtest_v02.py`: Direct import and call.
*   **Labeling (Indirect Consumer):**
    *   `app/ml/labels/reversal.py`: Consumes `ms_aggression_idx`.
    *   `app/ml/labels/vvh/reversal.py`: Consumes `ms_aggression_idx_v2`.

## 3. app.features.options_features

**Primary Function:** `compute_options_features`
**Key Outputs:** `opt_delta`, `opt_gamma`, `opt_gex`

### Consumers
*   **Production Pipeline:**
    *   `scripts/prepare_labels_dataset.py`: Imports and calls `compute_options_features`.
*   **Testing:**
    *   `scripts/test_labels.py`: Dynamic import and call.
    *   `scripts/test_all_features.py`: Direct import and call. Validates `opt_delta`, `opt_gamma`.
*   **Backtesting:**
    *   `app/orchestrator/backtest_v02.py`: Direct import and call.

## 4. Dataset Consumers (Merged Output)

These files consume the merged CSVs (`labels_dataset_v02_*.csv`) produced by the pipeline:
*   `scripts/validate_labels_dataset.py`: Reads and validates the datasets.
*   `scripts/train_smoke.py`: Consumes `X_train`, `X_val` for model training.
