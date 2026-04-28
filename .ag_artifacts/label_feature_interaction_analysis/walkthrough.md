# Interaction Analysis Walkthrough

This document summarizes the analysis of interactions between labeler modules (`cycle.py`, `regime.py`, `reversal.py`) and feature modules (`time_cycle.py`, `microstructure.py`, `options_features.py`).

## Generated Artifacts

All artifacts are located in this directory (`ag_artifacts/label_feature_interaction_analysis/`).

### 1. Interaction List
[interaction_list.csv](./interaction_list.csv)
A detailed CSV containing every identified interaction, including file path, line number, interaction type (import, call, reference), and code context.

### 2. Interaction Summary
[interaction_summary.md](./interaction_summary.md)
A human-readable report grouping interactions by producer module and consumer file.

### 3. Consumers Tree
[feature_and_label_consumers_tree.md](./feature_and_label_consumers_tree.md)
A hierarchical tree view showing which files consume which symbols from each producer module.

### 4. Dependency Graph
[feature_label_dependency_graph.md](./feature_label_dependency_graph.md)
A visual representation (Mermaid diagram) of the dependencies between modules and consumer files.

### 5. Missing References
- [missing_label_references.csv](./missing_label_references.csv)
- [missing_feature_references.csv](./missing_feature_references.csv)
These files list any missing references found during the scan. Currently, no missing references were identified.

## Key Findings

- **Core Consumers**: `app/orchestrator/backtest_v02.py` and `scripts/prepare_labels_dataset.py` are the primary consumers, interacting with all labeler and feature modules.
- **Testing**: `scripts/test_labels.py` and `tests/test_labels_integration.py` provide comprehensive coverage of the labeler modules.
- **Inter-Module Dependencies**: The labeler modules (`regime.py`, `reversal.py`) directly depend on feature columns produced by `time_cycle.py` and `microstructure.py`.
