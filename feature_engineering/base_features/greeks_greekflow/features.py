"""Deterministic computation of Greeks & Greek-Flow base features per snapshot."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from feature_engineering.utils.math_helpers import safe_divide


class FeatureComputationError(RuntimeError):
    """Raised when feature computation fails due to invalid inputs."""


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureComputationError(f"Missing required input columns: {missing}")


def _norm_cdf(x: pd.Series) -> pd.Series:
    return 0.5 * (1.0 + (x / math.sqrt(2.0)).apply(math.erf))


def _norm_pdf(x: pd.Series) -> pd.Series:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * np.power(x, 2))


def _compute_row_greeks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row Greek approximations using Black-Scholes style formulas."""
    working = df.copy()

    spot = working["spot"].astype(float)
    strike = working["strike"].astype(float)
    iv = working["implied_volatility"].astype(float)
    tte = working["time_to_expiry"].astype(float)

    # Guard against invalid inputs to avoid NaNs propagating unexpectedly.
    valid = (spot > 0) & (strike > 0) & (iv > 0) & (tte > 0)
    sqrt_t = np.sqrt(tte.where(valid, np.nan))
    log_term = np.log(safe_divide(spot, strike, allow_zero_division=False))
    d1 = (log_term + 0.5 * np.power(iv, 2) * tte) / (iv * sqrt_t)
    d1 = d1.where(valid, np.nan)
    d2 = d1 - iv * sqrt_t

    pdf_d1 = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_minus_d1 = _norm_cdf(-d1)

    delta = cdf_d1.where(working["option_type"] == "call", -cdf_minus_d1)
    gamma = safe_divide(pdf_d1, spot * iv * sqrt_t, allow_zero_division=True, fill_value=0.0)
    vega = spot * pdf_d1 * sqrt_t
    theta = -safe_divide(spot * pdf_d1 * iv, 2 * sqrt_t, allow_zero_division=True, fill_value=0.0)
    vanna = spot * pdf_d1 * (1 - d1 / (iv * sqrt_t)).where(valid, 0.0)
    charm = -gamma * spot * iv * sqrt_t

    working["delta"] = delta.fillna(0.0)
    working["gamma"] = gamma.fillna(0.0)
    working["vega"] = vega.fillna(0.0)
    working["theta"] = theta.fillna(0.0)
    working["vanna"] = vanna.fillna(0.0)
    working["charm"] = charm.fillna(0.0)
    return working


def _skew_slope(group: pd.DataFrame) -> float:
    iv_by_strike = group.groupby("strike")["implied_volatility"].mean().dropna()
    if iv_by_strike.empty:
        return 0.0
    spot_value = float(group["spot"].iloc[0])
    strikes_sorted = iv_by_strike.index.to_series().sort_values()
    lower_strikes = strikes_sorted[strikes_sorted <= spot_value]
    upper_strikes = strikes_sorted[strikes_sorted >= spot_value]

    if lower_strikes.empty or upper_strikes.empty:
        return 0.0

    lower_strike = float(lower_strikes.iloc[-1])
    upper_strike = float(upper_strikes.iloc[0])
    if lower_strike == upper_strike:
        return 0.0

    lower_iv = float(iv_by_strike.loc[lower_strike])
    upper_iv = float(iv_by_strike.loc[upper_strike])
    return (upper_iv - lower_iv) / (upper_strike - lower_strike)


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    oi = group["open_interest"].fillna(0.0)
    spot_value = float(group["spot"].mean()) if not group["spot"].isna().all() else 0.0

    delta_net = float((group["delta"] * oi).sum())
    gamma_net = float((group["gamma"] * oi).sum())
    vega_net = float((group["vega"] * oi).sum())
    theta_net = float((group["theta"] * oi).sum())
    vanna_net = float((group["vanna"] * oi).sum())
    charm_net = float((group["charm"] * oi).sum())

    option_notional = float((group["option_price"].fillna(0.0) * oi).sum())
    gamma_per_notional_series = safe_divide(
        pd.Series([gamma_net]), pd.Series([option_notional]), allow_zero_division=True, fill_value=0.0
    )
    gamma_per_notional = float(gamma_per_notional_series.iloc[0])

    skew_slope_value = float(_skew_slope(group))
    vol_of_vol = float(group["implied_volatility"].std(ddof=0)) if not group["implied_volatility"].isna().all() else 0.0

    mean_iv = float(group["implied_volatility"].mean()) if not group["implied_volatility"].isna().all() else 0.0

    delta_hedge_pressure = delta_net * spot_value * 0.01

    return pd.Series(
        {
            "greek__delta_net": delta_net,
            "greek__gamma_net": gamma_net,
            "greek__vega_net": vega_net,
            "greek__theta_net": theta_net,
            "greek__vanna_net": vanna_net,
            "greek__charm_net": charm_net,
            "greek__skew_slope": skew_slope_value,
            "greek__vol_of_vol_proxy": vol_of_vol,
            "greek__delta_hedge_pressure": delta_hedge_pressure,
            "greek__gamma_per_notional": gamma_per_notional,
            "_mean_implied_volatility": mean_iv,
            "_mean_spot": spot_value,
        }
    )


def compute_greeks_greekflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Greeks & Greek-Flow base features using only current and previous snapshots."""

    required_columns = [
        "ts",
        "symbol",
        "spot",
        "option_type",
        "strike",
        "open_interest",
        "implied_volatility",
        "option_price",
        "time_to_expiry",
    ]
    _require_columns(df, required_columns)

    enriched = _compute_row_greeks(df)

    aggregated = (
        enriched.groupby(["symbol", "ts"], as_index=False)
        .apply(_aggregate_group)
        .reset_index(drop=True)
    )

    # Compute flows and indicators using previous snapshot only.
    aggregated.sort_values(["symbol", "ts"], inplace=True)
    aggregated["prev_delta_net"] = aggregated.groupby("symbol")["greek__delta_net"].shift(1)
    aggregated["prev_gamma_net"] = aggregated.groupby("symbol")["greek__gamma_net"].shift(1)
    aggregated["prev_gamma_flow"] = aggregated.groupby("symbol")["greek__gamma_net"].diff()
    aggregated["prev_mean_iv"] = aggregated.groupby("symbol")["_mean_implied_volatility"].shift(1)
    aggregated["prev_spot"] = aggregated.groupby("symbol")["_mean_spot"].shift(1)

    aggregated["greek__delta_flow"] = aggregated["greek__delta_net"] - aggregated["prev_delta_net"].fillna(0.0)
    aggregated["greek__gamma_flow"] = aggregated["greek__gamma_net"] - aggregated["prev_gamma_net"].fillna(0.0)

    iv_change = aggregated["_mean_implied_volatility"] - aggregated["prev_mean_iv"].fillna(aggregated["_mean_implied_volatility"])
    aggregated["greek__implied_vol_surface_flag"] = np.sign(iv_change).astype(np.int32)

    spot_change_pct = safe_divide(
        aggregated["_mean_spot"] - aggregated["prev_spot"].fillna(aggregated["_mean_spot"]),
        aggregated["prev_spot"].fillna(aggregated["_mean_spot"]),
        allow_zero_division=True,
        fill_value=0.0,
    ).abs()
    iv_change_abs = iv_change.abs()
    sticky_raw = 1.0 / (1.0 + safe_divide(iv_change_abs, spot_change_pct + 1e-6, allow_zero_division=True, fill_value=0.0))
    aggregated["greek__sticky_delta_indicator"] = sticky_raw.clip(0.0, 1.0)

    prev_gamma_flow = aggregated["prev_gamma_flow"].fillna(0.0).abs()
    baseline = prev_gamma_flow + aggregated["prev_gamma_net"].fillna(0.0).abs() * 0.1 + 1e-9
    aggregated["greek__gamma_shock_indicator"] = (aggregated["greek__gamma_flow"].abs() > baseline * 2).astype(bool)

    # Fill remaining deterministic defaults.
    aggregated["greek__vol_of_vol_proxy"] = aggregated["greek__vol_of_vol_proxy"].fillna(0.0)

    # Select and cast final columns.
    final_columns = [
        "symbol",
        "ts",
        "greek__delta_net",
        "greek__gamma_net",
        "greek__vega_net",
        "greek__theta_net",
        "greek__delta_flow",
        "greek__gamma_flow",
        "greek__implied_vol_surface_flag",
        "greek__vanna_net",
        "greek__charm_net",
        "greek__skew_slope",
        "greek__vol_of_vol_proxy",
        "greek__gamma_shock_indicator",
        "greek__sticky_delta_indicator",
        "greek__delta_hedge_pressure",
        "greek__gamma_per_notional",
    ]

    result = aggregated[final_columns].copy()
    float_cols = [col for col in final_columns if col not in {"symbol", "ts", "greek__gamma_shock_indicator", "greek__implied_vol_surface_flag"}]
    result[float_cols] = result[float_cols].astype(np.float32)
    result["greek__gamma_shock_indicator"] = result["greek__gamma_shock_indicator"].astype(bool)
    result["greek__implied_vol_surface_flag"] = result["greek__implied_vol_surface_flag"].astype(np.int32)
    return result


__all__ = ["compute_greeks_greekflow_features", "FeatureComputationError"]
