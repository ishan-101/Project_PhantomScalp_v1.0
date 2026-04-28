"""Base funding-rate proxy engine.

Builds an institutional funding proxy from observed market microstructure:
- aggressive trade imbalance
- orderflow imbalance / inventory pressure
- open-interest expansion pressure
- directional extension pressure
- perp-vs-fair proxy premium pressure
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FundingRateEngineError(Exception):
    """Raised when funding-rate base state inputs are invalid."""


def _safe_div(num: pd.Series, den: pd.Series | float, eps: float = 1e-9) -> pd.Series:
    den_series = den if isinstance(den, pd.Series) else pd.Series(den, index=num.index)
    return num / den_series.abs().clip(lower=eps)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=5).mean().shift(1)
    sd = series.rolling(window=window, min_periods=5).std(ddof=0).shift(1)
    z = (series - mu) / sd.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_timeline(trades_df: pd.DataFrame, orderflow_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    trades_req = ["meta__timestamp", "meta__sequence_id", "price", "size", "aggressor"]
    of_req = [
        "meta__timestamp",
        "meta__sequence_id",
        "event_type",
        "price",
        "size",
        "aggressor",
        "inventory_pressure",
    ]
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
    trade_impulse = np.where(etype.eq("trade"), of["event_aggressor"] * of["event_size"], 0.0)
    add_cancel_impulse = np.where(etype.eq("add"), of["event_size"], np.where(etype.eq("cancel"), -of["event_size"], 0.0))
    of["orderflow_impulse"] = trade_impulse + 0.25 * add_cancel_impulse + 0.1 * of["inventory_pressure"]

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

    oi_map = oi_df[["meta__timestamp", "meta__sequence_id", "fut__open_interest", "fut__oi_change", "fut__oi_zscore"]].copy()
    oi_map["meta__timestamp"] = pd.to_datetime(oi_map["meta__timestamp"], utc=True)
    oi_map["meta__sequence_id"] = pd.to_numeric(oi_map["meta__sequence_id"], errors="coerce").fillna(0).astype("int64")
    oi_map = oi_map.sort_values(by=["meta__timestamp", "meta__sequence_id"], kind="mergesort").reset_index(drop=True)
    oi_map = oi_map.rename(columns={"meta__sequence_id": "oi__sequence_id"})

    merged = pd.merge_asof(
        tl,
        oi_map,
        on="meta__timestamp",
        by=None,
        direction="backward",
        allow_exact_matches=True,
    )
    if "meta__sequence_id_x" in merged.columns and "meta__sequence_id" not in merged.columns:
        merged = merged.rename(columns={"meta__sequence_id_x": "meta__sequence_id"})

    # Use sequence-aware forward guard: if first rows precede OI start, backfill from earliest valid OI snapshot.
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
    trade_imb = _safe_div(
        signed_notional.rolling(window=64, min_periods=1).sum(),
        abs_notional.rolling(window=64, min_periods=1).sum() + 1e-9,
    ).clip(-1.0, 1.0)

    orderflow_impulse = pd.to_numeric(out["orderflow_impulse"], errors="coerce").fillna(0.0)
    flow_scale = orderflow_impulse.abs().rolling(window=96, min_periods=1).mean().replace(0.0, np.nan)
    orderflow_imb = _safe_div(orderflow_impulse.rolling(window=96, min_periods=1).mean(), flow_scale).clip(-3.0, 3.0) / 3.0

    oi_change = pd.to_numeric(out["fut__oi_change"], errors="coerce").fillna(0.0)
    oi_level = pd.to_numeric(out["fut__open_interest"], errors="coerce").fillna(0.0)
    oi_expand = _safe_div(oi_change, oi_level.rolling(64, min_periods=1).mean().replace(0.0, np.nan)).clip(-0.5, 0.5)

    directional_alignment = np.sign(ret) * np.sign(oi_expand)
    oi_expansion_pressure = (directional_alignment * oi_expand.abs()).clip(-0.5, 0.5)

    ret_z = _rolling_zscore(ret, window=128).clip(-5.0, 5.0)
    directional_extension = (ret_z * oi_expand.abs()).clip(-2.5, 2.5) / 2.5

    fair_proxy = price.ewm(span=240, adjust=False, min_periods=10).mean().bfill().fillna(price)
    premium_proxy = _safe_div(price - fair_proxy, fair_proxy.replace(0.0, np.nan)).clip(-0.02, 0.02) / 0.02

    # Composite crowding pressure (institutional proxy)
    funding_proxy = (
        0.32 * trade_imb
        + 0.24 * orderflow_imb
        + 0.20 * oi_expansion_pressure
        + 0.14 * directional_extension
        + 0.10 * premium_proxy
    )

    # Convert crowding proxy to bounded funding estimate (per event proxy, ~8h rate scale)
    bounded = np.tanh(funding_proxy * 1.8)
    funding_rate = (bounded * 0.0015).clip(-0.0015, 0.0015)

    out["fut__funding_rate"] = funding_rate.astype("float32")

    # Persist economically meaningful internals for downstream deterministic features.
    out["__funding_trade_imbalance"] = trade_imb.astype("float32")
    out["__funding_orderflow_imbalance"] = orderflow_imb.astype("float32")
    out["__funding_oi_expansion_pressure"] = oi_expansion_pressure.astype("float32")
    out["__funding_directional_extension"] = directional_extension.astype("float32")
    out["__funding_premium_proxy"] = premium_proxy.astype("float32")
    out["__funding_price_return"] = ret.astype("float32")

    # Enforce zero-null for produced columns
    produced_cols = [
        "fut__funding_rate",
        "__funding_trade_imbalance",
        "__funding_orderflow_imbalance",
        "__funding_oi_expansion_pressure",
        "__funding_directional_extension",
        "__funding_premium_proxy",
        "__funding_price_return",
    ]
    out[produced_cols] = out[produced_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out
