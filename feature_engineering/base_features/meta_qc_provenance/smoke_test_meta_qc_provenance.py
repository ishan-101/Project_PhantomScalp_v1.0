"""End-to-end smoke test for Meta / QC / Provenance base features."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering.base_features.meta_qc_provenance.features import (
    META_FEATURE_NAMES,
    compute_meta_qc_provenance_features,
)
from feature_engineering.base_features.meta_qc_provenance.validator import (
    FeatureValidationError,
    validate_meta_qc_provenance_features,
)


def _build_base_frame() -> pd.DataFrame:
    source_ts = pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="s")
    ingest_ts = source_ts + pd.to_timedelta([5, 12, 20, 40, 80], unit="ms")
    return pd.DataFrame(
        {
            "source_timestamp": source_ts,
            "ingest_timestamp": ingest_ts,
            "price__last": [100.0, 100.5, 101.0, 100.75, 100.25],
            "ohlcv__close": [100.0, 100.5, 101.0, 100.75, 100.25],
            "volume__tick": [10.0, 12.0, 8.0, 15.0, 11.0],
            "source_feed": ["L1"] * 5,
        }
    )


def main() -> None:
    base_df = _build_base_frame()
    base_snapshot = base_df.copy(deep=True)

    enriched = compute_meta_qc_provenance_features(base_df)
    diagnostics = validate_meta_qc_provenance_features(enriched)

    new_columns = set(enriched.columns) - set(base_df.columns)
    if new_columns != set(META_FEATURE_NAMES):
        raise AssertionError(f"Expected exactly eight meta columns; found {sorted(new_columns)}")

    pd.testing.assert_frame_equal(base_df, base_snapshot, check_dtype=True)

    for col in ("meta__feature_confidence", "meta__ingest_quality_score"):
        values = enriched[col]
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise AssertionError(f"{col} contains values outside [0, 1]")

    corrupt = enriched.copy()
    corrupt.loc[0, "meta__feature_confidence"] = 2.0
    try:
        validate_meta_qc_provenance_features(corrupt)
    except FeatureValidationError:
        corruption_detected = True
    else:
        corruption_detected = False

    if not corruption_detected:
        raise AssertionError("Validator did not reject corrupted meta__feature_confidence.")

    assert diagnostics["validated"] is True
    print("meta_qc_provenance base features — SAFE TO FREEZE")
    print("Meta / QC / Provenance schema frozen; no upstream feature columns were modified.")


if __name__ == "__main__":
    main()
