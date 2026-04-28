"""Public utilities for feature engineering infrastructure."""

from .dtype_enforcement import DtypeValidationError, enforce_dtypes, validate_dtypes
from .math_helpers import log_returns, safe_divide, simple_difference, slope
from .null_handling import NullDiagnostics, NullHandlingStrategy, apply_null_handling
from .rolling import rolling_max, rolling_mean, rolling_min, rolling_std
from .validation_helpers import (
    ValidationError,
    check_duplicates,
    check_monotonic,
    check_shape,
    check_value_range,
)

__all__ = [
    "DtypeValidationError",
    "ValidationError",
    "NullHandlingStrategy",
    "NullDiagnostics",
    "apply_null_handling",
    "validate_dtypes",
    "enforce_dtypes",
    "log_returns",
    "simple_difference",
    "slope",
    "safe_divide",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "check_value_range",
    "check_monotonic",
    "check_shape",
    "check_duplicates",
]
