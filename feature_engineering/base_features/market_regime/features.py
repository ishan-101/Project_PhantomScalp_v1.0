import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when market regime feature computation fails."""


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _assert_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if df[col].isna().any():
            raise FeatureComputationError(f"Null values detected in required column '{col}'.")


def compute_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic market regime / volatility / liquidity features per snapshot."""

    required_columns = [
        "price__micro_volatility",
        "opt__best_bid_iv",
        "opt__best_ask_iv",
        "spread__l1",
        "previous_spread__l1",
        "ob__top_level_size_bid",
        "ob__top_level_size_ask",
        "ob__total_depth_bid",
        "ob__total_depth_ask",
        "ob__cancellation_rate",
        "ob__new_order_rate",
        "ob__top_of_book_turnover",
        "ob__queue_resilience",
        "ob__depth_elasticity",
        "ob__hidden_to_visible_ratio",
        "of__signed_volume",
        "previous_of__signed_volume",
        "of__aggressor_volume_ratio",
        "of__execution_flow_polarity",
        "of__trade_burst_intensity",
        "price__near_term_return_volatility",
        "price__tick_direction",
        "tick_return",
        "previous_tick_return",
        "previous_regime__realized_vol",
        "previous_ob__top_of_book_turnover",
        "previous_ob__queue_resilience",
    ]
    _require_columns(df, required_columns)
    _assert_no_nulls(df, required_columns)

    output = df.copy()

    spread = output["spread__l1"].astype("float32")
    prev_spread = output["previous_spread__l1"].astype("float32")

    # 1. Realized volatility (snapshot-safe)
    realized_vol = output["price__micro_volatility"].astype("float32")
    output["regime__realized_vol"] = realized_vol

    # 2. ATM implied volatility as mid of best bid/ask implied vols
    iv_bid = output["opt__best_bid_iv"].astype("float32")
    iv_ask = output["opt__best_ask_iv"].astype("float32")
    atm_iv = ((iv_bid + iv_ask) / 2.0).astype("float32")
    output["regime__iv_atm"] = atm_iv

    # 3. Liquidity score from spread tightness, top depth, and order update balance
    depth_top = (output["ob__top_level_size_bid"].astype(float) + output["ob__top_level_size_ask"].astype(float))
    depth_total = (output["ob__total_depth_bid"].astype(float) + output["ob__total_depth_ask"].astype(float))
    depth_ratio = safe_divide(pd.Series(depth_top.values), pd.Series(depth_total.values), allow_zero_division=True, fill_value=0.0)
    depth_ratio = depth_ratio.clip(lower=0.0, upper=1.0)
    spread_component = (1.0 / (1.0 + spread.replace(0, np.nan))).fillna(1.0)
    update_balance = 1.0 - (output["ob__cancellation_rate"].astype(float) / (output["ob__new_order_rate"].astype(float) + 1e-6))
    update_balance = update_balance.clip(lower=0.0, upper=1.0)
    liquidity_score = (0.4 * spread_component + 0.4 * depth_ratio + 0.2 * update_balance).astype("float32")
    liquidity_score = liquidity_score.clip(lower=0.0, upper=1.0)
    output["regime__liquidity_score"] = liquidity_score

    # 4. Volatility spike flag
    prev_realized = output["previous_regime__realized_vol"].astype(float)
    vol_spike_flag = (realized_vol > prev_realized * 1.5).astype(bool)
    output["regime__volatility_spike_flag"] = vol_spike_flag

    # 5. Spread stability
    stability = 1.0 - (spread - prev_spread).abs() / (prev_spread.abs() + 1e-6)
    stability = stability.clip(lower=0.0, upper=1.0).astype("float32")
    output["regime__spread_stability"] = stability

    # 6. Orderflow extremeness (z-scored signed volume intensity)
    signed_volume = output["of__signed_volume"].astype(float)
    prev_signed = output["previous_of__signed_volume"].astype(float)
    volume_z = safe_divide(pd.Series((signed_volume - prev_signed).values), pd.Series((prev_signed.abs() + 1e-6).values), allow_zero_division=True, fill_value=0.0)
    extremeness = (volume_z * (1.0 + output["of__aggressor_volume_ratio"].astype(float))).astype("float32")
    output["regime__orderflow_extremeness"] = extremeness

    # 7. Market state flag based on directionality and flow polarity
    tick_ret = output["tick_return"].astype(float)
    prev_tick_ret = output["previous_tick_return"].astype(float)
    flow_polarity = output["of__execution_flow_polarity"].astype(float)
    tick_dir = output["price__tick_direction"].astype(float)

    conditions_trend = (tick_ret > 0) & (prev_tick_ret > 0) & (flow_polarity > 0) & (tick_dir > 0)
    conditions_mean_revert = (tick_ret * prev_tick_ret < 0) | (flow_polarity < -0.5)
    market_state = np.select(
        [conditions_trend, conditions_mean_revert],
        ["trend", "mean_revert"],
        default="range",
    )
    output["regime__market_state_flag"] = pd.Series(market_state, index=df.index, dtype="category")

    # 8. Spread z-score relative to previous snapshot
    spread_z = safe_divide(pd.Series((spread - prev_spread).values), pd.Series((prev_spread.abs() + 1e-6).values), allow_zero_division=True, fill_value=0.0)
    output["regime__spread_zscore"] = spread_z.astype("float32")

    # 9. Liquidity fractal index using turnover and resilience irregularity
    turnover = output["ob__top_of_book_turnover"].astype(float)
    prev_turnover = output["previous_ob__top_of_book_turnover"].astype(float)
    resilience = output["ob__queue_resilience"].astype(float)
    prev_resilience = output["previous_ob__queue_resilience"].astype(float)
    turnover_change = (turnover - prev_turnover).abs()
    resilience_change = (resilience - prev_resilience).abs()
    fractal_index = safe_divide(
        pd.Series((turnover_change + resilience_change).values),
        pd.Series((turnover.abs() + resilience.abs() + 1e-6).values),
        allow_zero_division=True,
        fill_value=0.0,
    ).clip(lower=0.0, upper=2.0)
    output["regime__liquidity_fractal_index"] = fractal_index.astype("float32")

    # 10. Depth stress ratio
    depth_stress = safe_divide(
        pd.Series(spread.values),
        pd.Series((depth_total + 1e-6).values),
        allow_zero_division=True,
        fill_value=0.0,
    )
    output["regime__depth_stress_ratio"] = depth_stress.astype("float32")

    # 11. Volatility compression flag
    vol_compression_flag = (realized_vol < prev_realized * 0.8).astype(bool)
    output["regime__volatility_compression_flag"] = vol_compression_flag

    # 12. Momentum ignition flag
    ignition_flag = (
        (output["of__trade_burst_intensity"].astype(float) > 1.0)
        & (spread < prev_spread)
        & (realized_vol > prev_realized)
    ).astype(bool)
    output["regime__momentum_ignition_flag"] = ignition_flag

    # 13. Spread regime crossover
    crossover_flag = ((spread > prev_spread * 1.25) | (spread < prev_spread * 0.8)).astype(bool)
    output["regime__spread_regime_crossover"] = crossover_flag

    # 14. Short-term vol forecast error
    forecast = output["price__near_term_return_volatility"].astype(float)
    output["regime__short_term_vol_forecast_error"] = (realized_vol - forecast).astype("float32")

    # 15. Micro liquidity index
    elasticity = output["ob__depth_elasticity"].astype(float)
    hidden_ratio = output["ob__hidden_to_visible_ratio"].astype(float)
    depth_quality = safe_divide(pd.Series(depth_top.values), pd.Series((spread + 1e-6).values), allow_zero_division=True, fill_value=0.0)
    micro_liquidity = (
        0.4 * (1.0 / (1.0 + spread))
        + 0.3 * np.tanh(elasticity)
        + 0.2 * (1.0 - hidden_ratio.clip(lower=0.0))
        + 0.1 * depth_quality.clip(lower=0.0)
    )
    micro_liquidity = micro_liquidity.clip(lower=0.0, upper=1.0).astype("float32")
    output["regime__micro_liquidity_index"] = micro_liquidity

    feature_columns = [
        "regime__realized_vol",
        "regime__iv_atm",
        "regime__liquidity_score",
        "regime__volatility_spike_flag",
        "regime__spread_stability",
        "regime__orderflow_extremeness",
        "regime__market_state_flag",
        "regime__spread_zscore",
        "regime__liquidity_fractal_index",
        "regime__depth_stress_ratio",
        "regime__volatility_compression_flag",
        "regime__momentum_ignition_flag",
        "regime__spread_regime_crossover",
        "regime__short_term_vol_forecast_error",
        "regime__micro_liquidity_index",
    ]

    return output[feature_columns]
