# Feature Consumers Tree

```
Project Root
├── app/features/time_cycle.py
│   ├── [Call/Import] scripts/prepare_labels_dataset.py
│   ├── [Call/Import] scripts/test_labels.py
│   ├── [Call/Import] scripts/test_all_features.py
│   ├── [Call/Import] app/orchestrator/backtest_v02.py
│   └── [Data Consumer] app/ml/labels/regime.py (tc_atr, tc_adx...)
│   └── [Data Consumer] app/ml/labels/reversal.py (tc_rsi_14)
│
├── app/features/microstructure.py
│   ├── [Call/Import] scripts/prepare_labels_dataset.py
│   ├── [Call/Import] scripts/test_labels.py
│   ├── [Call/Import] scripts/test_all_features.py
│   ├── [Call/Import] app/orchestrator/backtest_v02.py
│   └── [Data Consumer] app/ml/labels/reversal.py (ms_aggression_idx)
│
├── app/features/options_features.py
│   ├── [Call/Import] scripts/prepare_labels_dataset.py
│   ├── [Call/Import] scripts/test_labels.py
│   ├── [Call/Import] scripts/test_all_features.py
│   └── [Call/Import] app/orchestrator/backtest_v02.py
│
└── [Merged Dataset Consumers] (labels_dataset_v02_*.csv)
    ├── scripts/validate_labels_dataset.py
    └── scripts/train_smoke.py
```
