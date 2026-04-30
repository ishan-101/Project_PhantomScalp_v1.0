"""Volume burst intensity feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolumeBurstIntensityEngineError(Exception):
    pass


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window=window, min_periods=max(10, window // 5)).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_side(aggressor: pd.Series) -> pd.Series:
    side = aggressor.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "buy": 1,
        "b": 1,
        "buyer": 1,
        "-1": -1,
        "sell": -1,
        "s": -1,
        "seller": -1,
        "0": 0,
        "none": 0,
        "neutral": 0,
    }
    return side.map(mapping).fillna(0).astype("int8")


def add_volume_burst_intensity(
    df: pd.DataFrame,
    orderflow_df: pd.DataFrame,
    short_window: int = 25,
    long_window: int = 250,
) -> pd.DataFrame:
    """Detect aggressive-execution clustering with frequency, large-trade, and delta-acceleration terms."""
    required = [
        "meta__timestamp",
        "trade_size",
        "aggressor_side",
        "fut__large_trade_volume",
        "fut__volume_delta",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise VolumeBurstIntensityEngineError(f"Missing columns: {missing}")

    out = df.copy(deep=True)
    ts = pd.to_datetime(out["meta__timestamp"], utc=True)
    size = pd.to_numeric(out["trade_size"], errors="coerce")
    if size.isna().any() or (size <= 0).any():
        raise VolumeBurstIntensityEngineError("trade_size contains invalid values")

    # 1) orderflow-confirmed aggressive event burst rate near each executed trade.
    of = orderflow_df.copy(deep=True)
    of["meta__timestamp"] = pd.to_datetime(of["meta__timestamp"], utc=True)
    of["event_size"] = pd.to_numeric(of["event_size"], errors="coerce").fillna(0.0)
    of["norm_aggr"] = _normalize_side(of["aggressor_side"])
    of["is_aggressive_exec"] = ((of["event_type"].astype(str).str.lower() == "trade") & (of["norm_aggr"] != 0)).astype("int8")
    of["aggressive_exec_size"] = np.where(of["is_aggressive_exec"] == 1, of["event_size"], 0.0)
    of = of.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)
    of["cum_aggr_exec_count"] = of["is_aggressive_exec"].cumsum()
    of["cum_aggr_exec_size"] = of["aggressive_exec_size"].cumsum()

    aligned = pd.merge_asof(
        out.sort_values(["meta__timestamp", "meta__sequence_id"], kind="mergesort"),
        of[["meta__timestamp", "cum_aggr_exec_count", "cum_aggr_exec_size"]],
        on="meta__timestamp",
        direction="backward",
    )
    aggr_count_step = aligned["cum_aggr_exec_count"].diff().fillna(aligned["cum_aggr_exec_count"]).clip(lower=0.0)
    aggr_size_step = aligned["cum_aggr_exec_size"].diff().fillna(aligned["cum_aggr_exec_size"]).clip(lower=0.0)

    # 2) trade frequency acceleration (inverse inter-arrival time, normalized)
    interarrival_sec = ts.diff().dt.total_seconds().replace(0, np.nan).bfill().fillna(1e-3)
    freq = 1.0 / interarrival_sec.clip(lower=1e-6)
    freq_z = _rolling_zscore(freq.astype("float64"), long_window)

    aggr_rate = aggr_count_step.rolling(window=short_window, min_periods=5).sum() / short_window
    aggr_size_share = (
        aggr_size_step.rolling(window=short_window, min_periods=5).sum()
        / size.rolling(window=short_window, min_periods=5).sum().replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aggr_composite_z = _rolling_zscore((aggr_rate + aggr_size_share).astype("float64"), long_window)

    # 3) large-trade concentration: short-vs-long participation share
    large_cum = pd.to_numeric(out["fut__large_trade_volume"], errors="coerce").astype("float64")
    large_inc = large_cum.diff().fillna(large_cum).clip(lower=0.0)
    total_short = size.rolling(window=short_window, min_periods=5).sum()
    large_short = large_inc.rolling(window=short_window, min_periods=5).sum()
    large_concentration = (large_short / total_short.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    large_concentration_z = _rolling_zscore(large_concentration, long_window)

    # 4) directional aggression acceleration from second derivative of validated delta
    delta = pd.to_numeric(out["fut__volume_delta"], errors="coerce").astype("float64")
    delta_velocity = delta.diff().fillna(0.0)
    delta_accel = delta_velocity.diff().fillna(0.0).abs()
    delta_accel_z = _rolling_zscore(delta_accel, long_window)

    burst = (0.30 * freq_z) + (0.30 * large_concentration_z) + (0.20 * delta_accel_z) + (0.20 * aggr_composite_z)
    burst = burst.clip(lower=0.0)
    out["fut__volume_burst_intensity"] = burst.astype("float32")

    return out
