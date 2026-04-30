# open_interest/features.py (FINAL CORRECTED)

from __future__ import annotations

import numpy as np
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
    "oi": "fut__open_interest__mtf-none__strike-none__maturity-none",
    "oi_change": "fut__oi_change__mtf-none__strike-none__maturity-none",
    "oi_velocity": "fut__oi_velocity__mtf-none__strike-none__maturity-none",
    "oi_acceleration": "fut__oi_acceleration__mtf-none__strike-none__maturity-none",
    "oi_zscore": "fut__oi_zscore__mtf-none__strike-none__maturity-none",
    "oi_price_div": "fut__oi_price_divergence__mtf-none__strike-none__maturity-none",
    "oi_price_div_strength": "fut__oi_price_divergence_strength__mtf-none__strike-none__maturity-none",
    "oi_turnover": "fut__oi_turnover__mtf-none__strike-none__maturity-none",
    "oi_open_close_ratio": "fut__oi_open_close_ratio__mtf-none__strike-none__maturity-none",
}


def compute_features(snapshot: pd.DataFrame, config: dict) -> pd.DataFrame:

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------
    required = ["open_interest", "price", "volume"]
    missing = [c for c in required if c not in snapshot.columns]
    if missing:
        raise ValueError(f"[open_interest] Missing columns: {missing}")

    df = pd.DataFrame(index=snapshot.index)

    oi = snapshot["open_interest"]
    price = snapshot["price"]
    volume = snapshot["volume"]

    window = config.get("rolling_window", 50)

    # ------------------------------------------------------------
    # 1. Base
    # ------------------------------------------------------------
    df[FEATURE_NAMES["oi"]] = oi
    df[[FEATURE_NAMES["oi"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi"]]],
        NullHandlingStrategy.FORWARD_FILL_THEN_ZERO
    )

    # ------------------------------------------------------------
    # 2. Change
    # ------------------------------------------------------------
    oi_change = simple_difference(oi)
    df[FEATURE_NAMES["oi_change"]] = oi_change
    df[[FEATURE_NAMES["oi_change"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_change"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 3. Velocity
    # ------------------------------------------------------------
    oi_velocity = slope(oi, periods=1)
    df[FEATURE_NAMES["oi_velocity"]] = oi_velocity
    df[[FEATURE_NAMES["oi_velocity"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_velocity"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 4. Acceleration
    # ------------------------------------------------------------
    oi_acceleration = slope(oi_velocity, periods=1)
    df[FEATURE_NAMES["oi_acceleration"]] = oi_acceleration
    df[[FEATURE_NAMES["oi_acceleration"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_acceleration"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 5. Z-Score (FIXED)
    # ------------------------------------------------------------
    mean = rolling_mean(oi, window)
    std = rolling_std(oi, window)

    # critical fix
    std = std.replace(0, pd.NA)

    oi_z = safe_divide_zero_safe(oi - mean, std)

    df[FEATURE_NAMES["oi_zscore"]] = oi_z
    df[[FEATURE_NAMES["oi_zscore"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_zscore"]]],
        NullHandlingStrategy.COMPUTE_USING_BASELINE_STATS_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 6. Divergence (FIXED)
    # ------------------------------------------------------------
    price_ret = simple_difference(price)

    std_oi = rolling_std(oi_change, window).replace(0, pd.NA)
    std_price = rolling_std(price_ret, window).replace(0, pd.NA)

    norm_oi = safe_divide_zero_safe(oi_change, std_oi)
    norm_price = safe_divide_zero_safe(price_ret, std_price)

    div = norm_oi - norm_price

    df[FEATURE_NAMES["oi_price_div"]] = div
    df[[FEATURE_NAMES["oi_price_div"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_price_div"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 7. Divergence Strength
    # ------------------------------------------------------------
    div_strength = np.abs(norm_oi - norm_price)

    df[FEATURE_NAMES["oi_price_div_strength"]] = div_strength
    df[[FEATURE_NAMES["oi_price_div_strength"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_price_div_strength"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 8. Turnover
    # ------------------------------------------------------------
    turnover = safe_divide_zero_safe(volume, oi)

    df[FEATURE_NAMES["oi_turnover"]] = turnover
    df[[FEATURE_NAMES["oi_turnover"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_turnover"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ZERO
    )

    # ------------------------------------------------------------
    # 9. Open/Close Ratio
    # ------------------------------------------------------------
    open_part = np.maximum(oi_change, 0)
    close_part = np.abs(np.minimum(oi_change, 0))

    ratio = safe_divide_zero_safe(open_part, close_part)
    ratio = ratio.replace(0, 1.0)

    df[FEATURE_NAMES["oi_open_close_ratio"]] = ratio
    df[[FEATURE_NAMES["oi_open_close_ratio"]]], _ = apply_null_handling(
        df[[FEATURE_NAMES["oi_open_close_ratio"]]],
        NullHandlingStrategy.COMPUTE_ELSE_ONE
    )

    # ------------------------------------------------------------
    # FINAL DTYPE ENFORCEMENT (SAFE)
    # ------------------------------------------------------------
    for col in df.columns:
        df[col] = df[col].astype("float32")

    return df
