# funding_rate/dependencies.py (FINAL CORRECTED)

from __future__ import annotations

from typing import List


REQUIRES_RAW: List[str] = [
    "funding_rate",
]


REQUIRES_FEATURES: List[str] = [
    "fut__oi_zscore__mtf-none__strike-none__maturity-none",
]


PROVIDES_FEATURES: List[str] = [
    "fut__funding_rate__mtf-none__strike-none__maturity-none",
    "fut__funding_rate_change__mtf-none__strike-none__maturity-none",
    "fut__funding_rate_velocity__mtf-none__strike-none__maturity-none",
    "fut__funding_rate_acceleration__mtf-none__strike-none__maturity-none",
    "fut__funding_rate_zscore__mtf-none__strike-none__maturity-none",
    "fut__funding_pressure_index__mtf-none__strike-none__maturity-none",
    "fut__funding_extreme_flag__mtf-none__strike-none__maturity-none",
    "fut__funding_oi_stress__mtf-none__strike-none__maturity-none",
    "fut__funding_rate_regime_flag__mtf-none__strike-none__maturity-none",
]


FEATURE_FAMILY: str = "futures"
SUBFAMILY: str = "funding_rate"
VERSION: str = "1.0"


def validate_raw_inputs(columns: List[str]) -> None:
    missing = [col for col in REQUIRES_RAW if col not in columns]
    if missing:
        raise ValueError(f"[funding_rate] Missing raw inputs: {missing}")


def validate_feature_outputs(features: List[str]) -> None:
    missing = [f for f in PROVIDES_FEATURES if f not in features]
    if missing:
        raise ValueError(f"[funding_rate] Missing features: {missing}")
