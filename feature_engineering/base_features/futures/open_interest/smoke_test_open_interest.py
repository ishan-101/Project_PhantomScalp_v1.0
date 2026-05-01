"""Smoke test for open-interest feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import compute_features
from validator import SCHEMA_DTYPES, validate_features


def run_smoke_test() -> None:
    idx = pd.date_range("2025-01-01", periods=240, freq="min", tz="UTC")
    rng = np.random.default_rng(7)

    snapshot = pd.DataFrame(
        {
            "open_interest": np.abs(10000 + np.cumsum(rng.normal(0, 15, len(idx)))).astype("float64"),
            "price": np.abs(2000 + np.cumsum(rng.normal(0, 2, len(idx)))).astype("float64"),
            "volume": np.abs(rng.normal(350, 100, len(idx))).astype("float64"),
        },
        index=idx,
    )

    config = {"rolling_window": 50}
    run_a = compute_features(snapshot, pd.DataFrame(index=idx), config)
    run_b = compute_features(snapshot, pd.DataFrame(index=idx), config)

    if run_a.isna().any().any():
        raise AssertionError("Smoke test failed: nulls detected")

    for col, dtype in SCHEMA_DTYPES.items():
        if str(run_a[col].dtype) != dtype:
            raise AssertionError(f"Smoke test failed: dtype mismatch for {col}")

    if not run_a.equals(run_b):
        raise AssertionError("Smoke test failed: output is not deterministic")

    validate_features(run_a)


if __name__ == "__main__":
    run_smoke_test()
    print("Open-interest smoke test passed.")
