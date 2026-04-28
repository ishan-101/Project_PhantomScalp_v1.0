"""Data Access Layer public interface (v0.1).

This package converts raw parquet inputs into schema-aligned, validated
DataFrames suitable for downstream feature engineering.
"""

from .aligner import AlignmentError, align_feeds_on_timestamp, sort_and_validate_timestamp
from .dtype_enforcer import DtypeEnforcementError, enforce_dtypes
from .parquet_loader import load_parquet
from .raw_schema import FieldSpec, FeedSchema, RAW_SCHEMAS
from .validator import ValidationError, validate

__all__ = [
    "AlignmentError",
    "align_feeds_on_timestamp",
    "sort_and_validate_timestamp",
    "DtypeEnforcementError",
    "enforce_dtypes",
    "load_parquet",
    "FieldSpec",
    "FeedSchema",
    "RAW_SCHEMAS",
    "ValidationError",
    "validate",
]
