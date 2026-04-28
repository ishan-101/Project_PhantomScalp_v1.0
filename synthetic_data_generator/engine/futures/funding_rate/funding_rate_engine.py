"""Institutional-grade funding rate proxy engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FundingRateEngineError(Exception):
    """Raised when required funding source inputs are invalid."""


def _safe_div(num: pd.Series, den: pd.Series | float, eps: float = 1e-9) -> pd.Series:
    den_series = den if isinstance(den, pd.Series) else pd.Series(den, index=num.index)
    return num / den_series.abs().clip(lower=eps)


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
    sd = series.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)
    z = (series - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _minutes_to_next_settlement(ts: pd.Series) -> pd.Series:
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute + (ts.dt.second / 60.0)
    slots = np.array([0.0, 480.0, 960.0, 1440.0], dtype=np.float64)  # 00:00, 08:00, 16:00, next day
    arr = minutes_of_day.to_numpy(dtype=np.float64)
    idx = np.searchsorted(slots, arr, side="right")
    next_slots = slots[idx]
    return pd.Series(next_slots - arr, index=ts.index, dtype="float64")


def _settlement_decay_weight(minutes_to_settlement: pd.Series) -> pd.Series:
    # Peaks at 20 minutes before settlement, remains elevated in the final 15-30 minute window.
    centered = ((minutes_to_settlement - 20.0) / 18.0) ** 2
    weight = np.exp(-centered)
    return pd.Series(weight, index=minutes_to_settlement.index, dtype="float64").clip(0.05, 1.0)


def _build_timeline(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    trades_req = ["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]
    of_req = ["meta__timestamp", "meta__sequence_id", "event_type", "price", "size", "aggressor", "inventory_pressure"]
    oi_req = ["meta__timestamp", "meta__sequence_id", "fut__open_interest", "fut__oi_change", "fut__oi_zscore"]

    missing_tr = [c for c in trades_req if c not in trades_df.columns]
    missing_of = [c for c in of_req if c not in orderflow_df.columns]
    missing_oi = [c for c in oi_req if c not in oi_df.columns]
    if missing_tr:
        raise FundingRateEngineError(f"trades_df missing required columns: {missing_tr}")
    if missing_of:
        raise FundingRateEngineError(f"orderflow_df missing required columns: {missing_of}")
    if missing_oi:
        raise FundingRateEngineError(f"oi_df missing required columns: {missing_oi}")

    tr = trades_df[trades_req].copy()
    tr["event_price"] = pd.to_numeric(tr["price"], errors="coerce").fillna(0.0)
    tr["event_size"] = pd.to_numeric(tr["size"], errors="coerce").fillna(0.0)
    tr["event_aggressor"] = pd.to_numeric(tr["aggressor"], errors="coerce").fillna(0.0)
    tr["trade_notional_signed"] = tr["event_price"] * tr["event_size"] * tr["event_aggressor"]
    tr["trade_notional_total"] = tr["event_price"] * tr["event_size"]
    tr["orderflow_impulse"] = 0.0
    tr["inventory_pressure"] = 0.0

    of = orderflow_df[of_req].copy()
    of["event_price"] = pd.to_numeric(of["price"], errors="coerce").fillna(0.0)
    of["event_size"] = pd.to_numeric(of["size"], errors="coerce").fillna(0.0)
    of["event_aggressor"] = pd.to_numeric(of["aggressor"], errors="coerce").fillna(0.0)
    of["inventory_pressure"] = pd.to_numeric(of["inventory_pressure"], errors="coerce").fillna(0.0)

    etype = of["event_type"].astype(str).str.lower()
    aggressive_flow = np.where(etype.eq("trade"), of["event_aggressor"] * of["event_size"], 0.0)
    queue_pressure = np.where(etype.eq("add"), of["event_size"], np.where(etype.eq("cancel"), -of["event_size"], 0.0))
    of["orderflow_impulse"] = aggressive_flow + 0.20 * queue_pressure + 0.15 * of["inventory_pressure"]

    of["trade_notional_signed"] = 0.0
    of["trade_notional_total"] = 0.0

    tl_cols = [
        "meta__timestamp",
        "meta__sequence_id",
        "event_price",
        "event_size",
        "event_aggressor",
        "trade_notional_signed",
        "trade_notional_total",
        "orderflow_impulse",
        "inventory_pressure",
    ]

    tl = pd.concat([tr[tl_cols], of[tl_cols]], ignore_index=True)
    tl["meta__timestamp"] = pd.to_datetime(tl["meta__timestamp"], utc=True)
    tl["meta__sequence_id"] = pd.to_numeric(tl["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    tl = tl.sort_values(by=["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    oi_map = oi_df[oi_req].copy()
    oi_map["meta__timestamp"] = pd.to_datetime(oi_map["meta__timestamp"], utc=True)
    oi_map["meta__sequence_id"] = pd.to_numeric(oi_map["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    oi_map = oi_map.sort_values(by=["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(tl, oi_map, on="meta__timestamp", direction="backward", allow_exact_matches=True)
    if "meta__sequence_id_x" in merged.columns:
        merged = merged.rename(columns={"meta__sequence_id_x": "meta__sequence_id"})
    if "meta__sequence_id_y" in merged.columns:
        merged = merged.drop(columns=["meta__sequence_id_y"])

    for col in ["fut__open_interest", "fut__oi_change", "fut__oi_zscore"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill().bfill().fillna(0.0)

    if merged[["meta__timestamp", "meta__sequence_id"]].isna().any().any():
        raise FundingRateEngineError("timeline contains null meta columns")

    return merged


def add_funding_rate(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    out = _build_timeline(trades_df, orderflow_df, oi_df)

    price = pd.to_numeric(out["event_price"], errors="coerce").fillna(0.0)
    ret = price.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    signed_notional = pd.to_numeric(out["trade_notional_signed"], errors="coerce").fillna(0.0)
    abs_notional = pd.to_numeric(out["trade_notional_total"], errors="coerce").fillna(0.0)
    aggressive_imbalance = _safe_div(
        signed_notional.rolling(window=96, min_periods=1).sum(),
        abs_notional.rolling(window=96, min_periods=1).sum() + 1e-9,
    ).clip(-1.0, 1.0)

    flow = pd.to_numeric(out["orderflow_impulse"], errors="coerce").fillna(0.0)
    flow_scale = flow.abs().rolling(window=128, min_periods=1).mean().replace(0.0, np.nan)
    orderflow_crowding = _safe_div(flow.rolling(window=128, min_periods=1).mean(), flow_scale).clip(-4.0, 4.0) / 4.0

    oi_change = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0)
    oi_level = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)
    oi_z = pd.to_numeric(out["fut__oi_zscore"], errors="coerce").fillna(0.0)
    oi_growth = _safe_div(oi_change, oi_level.rolling(window=96, min_periods=1).mean().replace(0.0, np.nan)).clip(-0.6, 0.6)

    trend_alignment = np.sign(ret) * np.sign(oi_growth)
    leverage_crowding = (0.60 * oi_growth + 0.40 * (oi_z.clip(-4.0, 4.0) / 4.0)) * trend_alignment

    fair_proxy = price.ewm(span=300, adjust=False, min_periods=20).mean().bfill().fillna(price)
    premium_pressure = _safe_div(price - fair_proxy, fair_proxy.replace(0.0, np.nan)).clip(-0.02, 0.02) / 0.02

    minutes_to_settlement = _minutes_to_next_settlement(out["meta__timestamp"])
    settlement_weight = _settlement_decay_weight(minutes_to_settlement)

    raw_crowding = (
        0.34 * aggressive_imbalance
        + 0.24 * orderflow_crowding
        + 0.24 * leverage_crowding
        + 0.18 * premium_pressure
    )

    # Funding relevance rises toward settlement windows.
    settlement_normalized = raw_crowding * (0.70 + 0.30 * settlement_weight)
    funding_rate = np.tanh(settlement_normalized * 2.1) * 0.0015

    out["fut__funding_rate"] = pd.Series(funding_rate, index=out.index).clip(-0.0015, 0.0015).astype("float32")
    out["__minutes_to_settlement"] = minutes_to_settlement.astype("float32")
    out["__funding_settlement_decay_weight"] = settlement_weight.astype("float32")

    out[["fut__funding_rate", "__minutes_to_settlement", "__funding_settlement_decay_weight"]] = (
        out[["fut__funding_rate", "__minutes_to_settlement", "__funding_settlement_decay_weight"]]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    return out
