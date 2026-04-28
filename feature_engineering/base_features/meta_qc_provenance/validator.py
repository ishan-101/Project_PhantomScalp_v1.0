"""Validator for Meta / QC / Provenance base features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import DtypeValidationError, validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range


class FeatureValidationError(ValueError):
    """Raised when Meta / QC / Provenance validation fails."""


META_COLUMNS = (
    "meta__timestamp",
    "meta__sequence_id",
    "meta__source_feed",
    "meta__data_latency_ms",
    "meta__feature_confidence",
    "meta__staleness_flag",
    "meta__ingest_quality_score",
    "meta__feature_mask_count",
)

ALLOWED_SOURCE_FEEDS = {"L1", "L2", "L3", "OPTIONS", "OTHER", "UNKNOWN"}


def _load_schema(schema_path: Path | None = None) -> List[Dict[str, Any]]:
    path = schema_path or Path(__file__).with_name("schema.json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["features"]


def _convert_bound(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "inf":
            return np.inf
        if value == "-inf":
            return -np.inf
    return float(value)


def _validate_timestamp(series: pd.Series) -> None:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValidationError("meta__timestamp contains nulls or non-ISO-8601 values.")
    if series.isna().any():
        raise ValidationError("meta__timestamp contains missing values.")


def _validate_sequence(series: pd.Series) -> None:
    if (series < 0).any():
        raise ValidationError("meta__sequence_id must be non-negative.")
    diffs = series.diff().dropna()
    if not (diffs >= 0).all():
        raise ValidationError("meta__sequence_id must be monotonically non-decreasing.")


def _validate_source_feed(series: pd.Series) -> None:
    if series.isna().any():
        raise ValidationError("meta__source_feed contains null values.")
    invalid = series[~series.isin(ALLOWED_SOURCE_FEEDS)]
    if not invalid.empty:
        raise ValidationError(
            f"meta__source_feed contains invalid entries at indices {list(invalid.index)}: "
            f"{invalid.unique().tolist()}"
        )


def _validate_ranges(df: pd.DataFrame, features: List[Dict[str, Any]]) -> None:
    for feature in features:
        name = feature["name"]
        if name in {"meta__timestamp", "meta__source_feed"}:
            continue
        lower_raw, upper_raw = feature.get("expected_range", (None, None))
        lower = _convert_bound(lower_raw)
        upper = _convert_bound(upper_raw)
        check_value_range(df[name], min_value=lower, max_value=upper, inclusive=True)


def _validate_boolean(series: pd.Series, name: str) -> None:
    if series.dtype != bool:
        raise DtypeValidationError(f"{name} must have dtype bool; found {series.dtype}.")
    if series.isna().any():
        raise ValidationError(f"{name} contains null values.")


def validate_meta_qc_provenance_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Meta / QC / Provenance base features according to the frozen schema."""

    schema_features = _load_schema()
    expected_columns = [feature["name"] for feature in schema_features]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise FeatureValidationError(f"Missing meta/QC columns: {missing}")

    dtype_expectations = {feature["name"]: feature["dtype"] for feature in schema_features}
    try:
        validate_dtypes(df, dtype_expectations)
    except DtypeValidationError as exc:
        raise FeatureValidationError(str(exc)) from exc

    try:
        _validate_timestamp(df["meta__timestamp"])
        _validate_sequence(df["meta__sequence_id"])
        _validate_source_feed(df["meta__source_feed"])
        _validate_boolean(df["meta__staleness_flag"], "meta__staleness_flag")
        _validate_ranges(df, schema_features)
    except (ValidationError, DtypeValidationError) as exc:
        raise FeatureValidationError(str(exc)) from exc

    range_columns = ["meta__feature_confidence", "meta__ingest_quality_score"]
    for col in range_columns:
        values = df[col]
        if values.isna().any():
            raise FeatureValidationError(f"{col} contains null values.")
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise FeatureValidationError(f"{col} must lie within [0, 1].")

    null_counts = {col: int(df[col].isna().sum()) for col in expected_columns}
    if any(count > 0 for count in null_counts.values()):
        raise FeatureValidationError(
            "Nulls present in meta / QC / Provenance features: "
            + ", ".join(f"{col}={count}" for col, count in null_counts.items() if count > 0)
        )

    return {"validated": True, "null_counts": null_counts}


__all__ = ["validate_meta_qc_provenance_features", "FeatureValidationError"]
