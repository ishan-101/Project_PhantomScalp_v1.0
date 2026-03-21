# Feature and Label Consumers Tree

cycle
    ├── app/orchestrator/backtest_v02.py (label_cycle)
    ├── scripts/check_label_stats.py (labels_only*.csv)
    ├── scripts/prepare_labels_dataset.py (label_cycle)
    ├── scripts/test_labels.py (label_cycle)
    ├── scripts/tune_cycle_reversal.py (label_cycle)
    ├── tests/test_labels_integration.py (label_cycle)

microstructure
    ├── app/ml/labels/reversal.py (ms_absorption_flag, ms_aggression_idx, ms_micro_divergence, ms_microtrend_vector, ms_mid, ms_spread, ms_sweep_flag)
    ├── app/orchestrator/backtest_v02.py (compute_microstructure_features)
    ├── scripts/prepare_labels_dataset.py (compute_microstructure_features, microstructure)
    ├── scripts/test_all_features.py (compute_microstructure_features)
    ├── scripts/test_labels.py (microstructure)

options_features
    ├── app/orchestrator/backtest_v02.py (compute_options_features)
    ├── scripts/prepare_labels_dataset.py (compute_options_features, options_features)
    ├── scripts/test_all_features.py (compute_options_features)
    ├── scripts/test_labels.py (options_features)

regime
    ├── app/orchestrator/backtest_v02.py (label_regime)
    ├── scripts/check_label_stats.py (labels_only*.csv)
    ├── scripts/prepare_labels_dataset.py (label_regime, labels_dataset_v02_*.csv)
    ├── scripts/test_labels.py (label_regime)
    ├── scripts/train_smoke.py (*_train.parquet, regime_sig)
    ├── tests/test_labels_integration.py (label_regime)

reversal
    ├── app/orchestrator/backtest_v02.py (label_reversal)
    ├── scripts/check_label_stats.py (labels_only*.csv)
    ├── scripts/inspect_reversal_debug.py (labels_only_v02_*.csv, reversal_debug.csv)
    ├── scripts/prepare_labels_dataset.py (label_reversal)
    ├── scripts/show_accepted_reversals.py (labels_sample_v02.csv, reversal_debug.csv)
    ├── scripts/test_labels.py (label_reversal)
    ├── scripts/tune_cycle_reversal.py (label_reversal)
    ├── scripts/tune_reversal.py (label_reversal)
    ├── tests/test_labels_integration.py (label_reversal)

time_cycle
    ├── app/ml/labels/regime.py (tc_adx, tc_atr, tc_trend_strength, tc_vol_comp_ratio)
    ├── app/ml/labels/reversal.py (tc_atr, tc_rsi_14)
    ├── app/orchestrator/backtest_v02.py (compute_time_cycle_features, tc_atr)
    ├── scripts/prepare_labels_dataset.py (compute_time_cycle_features, time_cycle)
    ├── scripts/test_all_features.py (compute_time_cycle_features)
    ├── scripts/test_labels.py (time_cycle)

