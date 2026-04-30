# funding_rate/features.py (FINAL CORRECTED)

from __future__ import annotations

import pandas as pd

from feature_engineering.utils import (
    simple_difference,
    slope,
    rolling_mean,
    rolling_std,
    safe_divide_zero_safe,
    apply_null_handling,
    NullHandlingStrategy,
)

FEATURE_NAMES = {
    "funding_rate": "fut__funding_rate__mtf-none__strike-none__maturity-none",
    "funding_rate_change": "fut__funding_rate_change__mtf-none__strike-none__maturity-none",
    "funding_rate_velocity": "fut__funding_rate_velocity__mtf-none__strike-none__maturity-none",
    "funding_rate_acceleration": "fut__funding_rate_acceleration__mtf-none__strike-none__maturity-none",
    "funding_rate_zscore": "fut__funding_rate_zscore__mtf-none__strike-none__maturity-none",
    "funding_pressure_index": "fut__funding_pressure_index__mtf-none__strike-none__maturity-none",
    "funding_extreme_flag": "fut__funding_extreme_flag__mtf-none__strike-none__maturity-none",
    "funding_oi_stress": "fut__funding_oi_stress__mtf-none__strike-none__maturity-none",
    "funding_rate_regime_flag": "fut__funding_rate_regime_flag__mtf-none__strike-none__maturity-none",
}


def compute_features(snapshot: pd.DataFrame, upstream_features: pd.DataFrame, config: dict) -> pd.DataFrame:

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------
    required_snapshot = ["funding_rate"]
    missing = [c for c in required_snapshot if c not in snapshot.columns]
    if missing:
        raise ValueError(f"[funding_rate] Missing snapshot columns: {missing}")

    required_upstream = [
        "fut__oi_zscore__mtf-none__strike-none__maturity-none"
    ]
    missing = [c for c in required_upstream if c not in upstream_features.columns]
    if missing:
        raise ValueError(f"[funding_rate] Missing upstream features: {missing}")

    df = pd.DataFrame(index=snapshot.index)

    fr = snapshot["funding_rate"]
    oi_z = upstream_features["fut__oi_zscore__mtf-none__strike-none__maturity-none"]

    window = config.get("rolling_window", 50)

    # ------------------------------------------------------------
    # 1. Base
    # ------------------------------------------------------------
    df[FEATURE_NAMES["funding_rate"]] = fr
    df[[FEATURE_NAMES["funding_rate"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate"]]],
        NullHandlingStrategy.FORWARD_FILL_THEN_ZERO
    )

    # ------------------------------------------------------------
    # 2. Change
    # ------------------------------------------------------------
    fr_change = simple_difference(fr)
    df[FEATURE_NAMES["funding_rate_change"]] = fr_change
    df[[FEATURE_NAMES["funding_rate_change"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate_change"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 3. Velocity
    # ------------------------------------------------------------
    fr_velocity = slope(fr, periods=1)
    df[FEATURE_NAMES["funding_rate_velocity"]] = fr_velocity
    df[[FEATURE_NAMES["funding_rate_velocity"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate_velocity"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 4. Acceleration
    # ------------------------------------------------------------
    fr_acc = slope(fr_velocity, periods=1)
    df[FEATURE_NAMES["funding_rate_acceleration"]] = fr_acc
    df[[FEATURE_NAMES["funding_rate_acceleration"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate_acceleration"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 5. Z-Score (FIXED)
    # ------------------------------------------------------------
    mean = rolling_mean(fr, window)
    std = rolling_std(fr, window)

    std = std.replace(0, pd.NA)  # critical fix

    fr_z = safe_divide_zero_safe(fr - mean, std)

    df[FEATURE_NAMES["funding_rate_zscore"]] = fr_z
    df[[FEATURE_NAMES["funding_rate_zscore"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate_zscore"]]],
        NullHandlingStrategy.COMPUTE_USING_BASELINE_STATS_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 6. Funding Pressure Index (IMPROVED)
    # ------------------------------------------------------------
    pressure = fr_z * (1 + fr_velocity.abs())

    df[FEATURE_NAMES["funding_pressure_index"]] = pressure
    df[[FEATURE_NAMES["funding_pressure_index"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_pressure_index"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 7. Extreme Flag (FIXED DTYPE)
    # ------------------------------------------------------------
    extreme_flag = (fr_z.abs() > 2.0).astype("int32")

    df[FEATURE_NAMES["funding_extreme_flag"]] = extreme_flag
    df[[FEATURE_NAMES["funding_extreme_flag"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_extreme_flag"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 8. Funding OI Stress
    # ------------------------------------------------------------
    stress = fr_z * oi_z

    df[FEATURE_NAMES["funding_oi_stress"]] = stress
    df[[FEATURE_NAMES["funding_oi_stress"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_oi_stress"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 9. Regime Flag
    # ------------------------------------------------------------
    # TODO: improve regime using persistence/volatility later
    regime = (fr_z > 1.0).astype("int32") - (fr_z < -1.0).astype("int32")

    df[FEATURE_NAMES["funding_rate_regime_flag"]] = regime
    df[[FEATURE_NAMES["funding_rate_regime_flag"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["funding_rate_regime_flag"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # FINAL TYPE ENFORCEMENT
    # ------------------------------------------------------------
    for col in df.columns:
        if "flag" in col:
            df[col] = df[col].astype("int32")
        else:
            df[col] = df[col].astype("float32")

    return df
