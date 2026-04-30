# basis_structure/features.py (FINAL CORRECTED)

from __future__ import annotations

import pandas as pd

from feature_engineering.utils import (
    simple_difference,
    rolling_mean,
    rolling_std,
    safe_divide_zero_safe,
    apply_null_handling,
    NullHandlingStrategy,
)

FEATURE_NAMES = {
    "basis": "fut__basis__mtf-none__strike-none__maturity-none",
    "basis_change": "fut__basis_change__mtf-none__strike-none__maturity-none",
    "basis_zscore": "fut__basis_zscore__mtf-none__strike-none__maturity-none",
    "basis_trend": "fut__basis_trend__mtf-none__strike-none__maturity-none",
    "basis_volatility": "fut__basis_volatility__mtf-none__strike-none__maturity-none",
    "basis_regime_flag": "fut__basis_regime_flag__mtf-none__strike-none__maturity-none",
}


def compute_features(snapshot: pd.DataFrame, config: dict) -> pd.DataFrame:

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------
    required = ["perp_price", "spot_price"]
    missing = [c for c in required if c not in snapshot.columns]
    if missing:
        raise ValueError(f"[basis_structure] Missing columns: {missing}")

    df = pd.DataFrame(index=snapshot.index)

    perp_price = snapshot["perp_price"]
    spot_price = snapshot["spot_price"]

    window = config.get("rolling_window", 50)

    # ------------------------------------------------------------
    # 1. Basis (NORMALIZED - CRITICAL FIX)
    # ------------------------------------------------------------
    basis = safe_divide_zero_safe(perp_price - spot_price, spot_price)

    df[FEATURE_NAMES["basis"]] = basis
    df[[FEATURE_NAMES["basis"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 2. Basis change
    # ------------------------------------------------------------
    basis_change = simple_difference(basis)

    df[FEATURE_NAMES["basis_change"]] = basis_change
    df[[FEATURE_NAMES["basis_change"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis_change"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 3. Z-score (FIXED)
    # ------------------------------------------------------------
    mean = rolling_mean(basis, window)
    std = rolling_std(basis, window)

    std = std.replace(0, pd.NA)

    basis_z = safe_divide_zero_safe(basis - mean, std)

    df[FEATURE_NAMES["basis_zscore"]] = basis_z
    df[[FEATURE_NAMES["basis_zscore"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis_zscore"]]],
        NullHandlingStrategy.COMPUTE_USING_BASELINE_STATS_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 4. Trend (reuse mean)
    # ------------------------------------------------------------
    trend = mean

    df[FEATURE_NAMES["basis_trend"]] = trend
    df[[FEATURE_NAMES["basis_trend"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis_trend"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 5. Volatility
    # ------------------------------------------------------------
    vol = std

    df[FEATURE_NAMES["basis_volatility"]] = vol
    df[[FEATURE_NAMES["basis_volatility"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis_volatility"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 6. Regime flag
    # ------------------------------------------------------------
    # TODO: improve regime using volatility/persistence
    regime = (basis_z > 1).astype("int32") - (basis_z < -1).astype("int32")

    df[FEATURE_NAMES["basis_regime_flag"]] = regime
    df[[FEATURE_NAMES["basis_regime_flag"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["basis_regime_flag"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # FINAL DTYPE ENFORCEMENT
    # ------------------------------------------------------------
    for col in df.columns:
        if "flag" in col:
            df[col] = df[col].astype("int32")
        else:
            df[col] = df[col].astype("float32")

    return df
