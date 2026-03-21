"""Validation for orderflow_tick base features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from feature_engineering.utils.dtype_enforcement import validate_dtypes
from feature_engineering.utils.validation_helpers import ValidationError, check_value_range

SCHEMA_PATH = Path(__file__).with_name("schema.json")


def _load_schema() -> Dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_dtype_mapping(schema: Dict) -> Dict[str, str]:
    return {feature["name"]: feature["dtype"] for feature in schema.get("features", [])}


def _expected_ranges(schema: Dict) -> Dict[str, Tuple[float | None, float | None]]:
    ranges: Dict[str, Tuple[float | None, float | None]] = {}
    for feature in schema.get("features", []):
        raw_min, raw_max = feature.get("expected_range", [None, None])
        min_val = None if raw_min in (None, "-inf") else float(raw_min)
        max_val = None if raw_max in (None, "inf") else float(raw_max)
        ranges[feature["name"]] = (min_val, max_val)
    return ranges


def validate_orderflow_tick_features(df: pd.DataFrame) -> None:
    """Validate presence, dtypes, ranges, and null policies for orderflow features."""

    schema = _load_schema()
    feature_names = [feature["name"] for feature in schema.get("features", [])]

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing orderflow features: {missing}")

    expected_dtypes = _expected_dtype_mapping(schema)
    validate_dtypes(df, expected_dtypes)

    ranges = _expected_ranges(schema)
    for col, (min_val, max_val) in ranges.items():
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            if series.isna().any():
                raise ValidationError(f"Boolean feature '{col}' contains nulls.")
            continue
        check_value_range(series, min_value=min_val, max_value=max_val, inclusive=True)

    ratio_bounds = {
        "of__imbalance_ratio": (-1.0, 1.0),
        "of__aggressor_flag_ratio": (0.0, 1.0),
        "of__aggressor_volume_ratio": (0.0, None),
        "of__market_pressure_tilt": (-np.inf, np.inf),
        "of__realized_sign_rate": (0.0, 1.0),
        "of__initiator_persistence": (0.0, 1.0),
    }
    for col, (min_bound, max_bound) in ratio_bounds.items():
        series = df[col]
        check_value_range(series, min_value=min_bound, max_value=max_bound, inclusive=True)

    entropy_col = df.get("of__sequence_entropy")
    if entropy_col is not None:
        check_value_range(entropy_col, min_value=0.0, max_value=None, inclusive=True)

    nulls = {col: int(df[col].isna().sum()) for col in feature_names}
    lingering = {col: count for col, count in nulls.items() if count > 0}
    if lingering:
        raise ValidationError(f"Nulls present in validated features: {lingering}")

    run_columns = ["of__run_length_up", "of__run_length_down", "of__large_trade_count"]
    for col in run_columns:
        if (df[col] < 0).any():
            raise ValidationError(f"Negative run/count detected in '{col}'.")

    polarity_col = df.get("of__execution_flow_polarity")
    if polarity_col is not None:
        invalid = ~polarity_col.isin([-1, 0, 1])
        if invalid.any():
            raise ValidationError("execution_flow_polarity must be in {-1,0,1}.")

    aggression_persistence = df.get("of__aggression_persistence")
    if aggression_persistence is not None and aggression_persistence.isna().any():
        raise ValidationError("aggression_persistence contains nulls.")


FeatureValidationError = ValidationError
