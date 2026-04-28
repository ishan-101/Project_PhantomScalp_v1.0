# Interaction Summary

## Producer: cycle
- **app/orchestrator/backtest_v02.py**: Interacts via call, import referencing label_cycle
- **scripts/check_label_stats.py**: Interacts via file_read referencing labels_only*.csv
- **scripts/prepare_labels_dataset.py**: Interacts via call, import referencing label_cycle
- **scripts/test_labels.py**: Interacts via call, import referencing label_cycle
- **scripts/tune_cycle_reversal.py**: Interacts via call, import referencing label_cycle
- **tests/test_labels_integration.py**: Interacts via call, import referencing label_cycle

## Producer: microstructure
- **app/ml/labels/reversal.py**: Interacts via feature_reference referencing ms_absorption_flag, ms_aggression_idx, ms_micro_divergence, ms_microtrend_vector, ms_mid, ms_spread, ms_sweep_flag
- **app/orchestrator/backtest_v02.py**: Interacts via call, import referencing compute_microstructure_features
- **scripts/prepare_labels_dataset.py**: Interacts via call, import referencing compute_microstructure_features, microstructure
- **scripts/test_all_features.py**: Interacts via call, import referencing compute_microstructure_features
- **scripts/test_labels.py**: Interacts via import referencing microstructure

## Producer: options_features
- **app/orchestrator/backtest_v02.py**: Interacts via call, import referencing compute_options_features
- **scripts/prepare_labels_dataset.py**: Interacts via call, import referencing compute_options_features, options_features
- **scripts/test_all_features.py**: Interacts via call, import referencing compute_options_features
- **scripts/test_labels.py**: Interacts via import referencing options_features

## Producer: regime
- **app/orchestrator/backtest_v02.py**: Interacts via call, import referencing label_regime
- **scripts/check_label_stats.py**: Interacts via file_read referencing labels_only*.csv
- **scripts/prepare_labels_dataset.py**: Interacts via call, file_write, import referencing label_regime, labels_dataset_v02_*.csv
- **scripts/test_labels.py**: Interacts via call, import referencing label_regime
- **scripts/train_smoke.py**: Interacts via file_read, label_reference referencing *_train.parquet, regime_sig
- **tests/test_labels_integration.py**: Interacts via call, import referencing label_regime

## Producer: reversal
- **app/orchestrator/backtest_v02.py**: Interacts via call, import referencing label_reversal
- **scripts/check_label_stats.py**: Interacts via file_read referencing labels_only*.csv
- **scripts/inspect_reversal_debug.py**: Interacts via file_read referencing labels_only_v02_*.csv, reversal_debug.csv
- **scripts/prepare_labels_dataset.py**: Interacts via call, import referencing label_reversal
- **scripts/show_accepted_reversals.py**: Interacts via file_read referencing labels_sample_v02.csv, reversal_debug.csv
- **scripts/test_labels.py**: Interacts via call, import referencing label_reversal
- **scripts/tune_cycle_reversal.py**: Interacts via call, import referencing label_reversal
- **scripts/tune_reversal.py**: Interacts via call, import referencing label_reversal
- **tests/test_labels_integration.py**: Interacts via call, import referencing label_reversal

## Producer: time_cycle
- **app/ml/labels/regime.py**: Interacts via feature_reference referencing tc_adx, tc_atr, tc_trend_strength, tc_vol_comp_ratio
- **app/ml/labels/reversal.py**: Interacts via feature_reference referencing tc_atr, tc_rsi_14
- **app/orchestrator/backtest_v02.py**: Interacts via call, feature_reference, import referencing compute_time_cycle_features, tc_atr
- **scripts/prepare_labels_dataset.py**: Interacts via call, import referencing compute_time_cycle_features, time_cycle
- **scripts/test_all_features.py**: Interacts via call, import referencing compute_time_cycle_features
- **scripts/test_labels.py**: Interacts via import referencing time_cycle

