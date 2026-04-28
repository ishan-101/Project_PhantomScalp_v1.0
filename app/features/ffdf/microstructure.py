# app/features/microstructure.py
from __future__ import annotations
"""
microstructure.py - extended (v0.2) — patched to include:
  - ms_top_book_pressure
  - ms_orderflow_vel
  - ms_microtrend_vector
  - ms_heat_slope

Stable API:
    compute_microstructure_features(bar_df, trades_df=None, depth_df=None, params=None) -> DataFrame
"""
from typing import Optional, Dict
import numpy as np
import pandas as pd

DEFAULT_LEVELS = 5

FEATURE_MANIFEST = {
    "aggression_idx": "ms_aggression_idx",
    "l3_momentum_burst": "ms_l3_momentum_burst",
    "iceberg_flag": "ms_iceberg_flag",
    "sweep_flag": "ms_sweep_flag",
    "pull_stack_ratio": "ms_pull_stack_ratio",
    "exhaustion": "ms_exhaustion_score",
    "liq_decay": "ms_liq_decay",
    "vol_imb_lev": "ms_vol_imb_lev{}",
    "vol_imb_sum5": "ms_vol_imb_sum5",
    "micro_div": "ms_micro_divergence",
    "book_pressure": "ms_book_pressure_delta",
    "top_book_pressure": "ms_top_book_pressure",
    "imb_level": "ms_imb_level_{}",
    "heat_slope": "ms_heat_slope",
    "spoof_rate": "ms_spoof_rate",
    "absorption_flag": "ms_absorption_flag",
    "orderflow_vel": "ms_orderflow_vel",
    "microtrend_vector": "ms_microtrend_vector",
}

def _ensure_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("bar_df cannot be None")
    if "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bar_df must have a DatetimeIndex or a 'datetime' column")
    return df

def _spread_mid(df: pd.DataFrame, bid_col="bid", ask_col="ask"):
    if bid_col in df.columns and ask_col in df.columns:
        bid = df[bid_col].astype(float)
        ask = df[ask_col].astype(float)
        spread = (ask - bid).rename("ms_spread")
        mid = ((ask + bid) / 2.0).rename("ms_mid")
    else:
        mid = df["close"].astype(float).rename("ms_mid")
        spread = pd.Series(0.0, index=df.index).rename("ms_spread")
    spread_z = (spread - spread.rolling(60, min_periods=1).mean()) / spread.rolling(60, min_periods=1).std().replace(0, np.nan)
    return pd.concat([spread, mid, spread_z.rename("ms_spread_z")], axis=1)

def trade_agg_features(trades_df: pd.DataFrame, resample_rule: str = "1min"):
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame()
    df = trades_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    def side_to_num(x):
        s = str(x).lower()
        if s in ("b", "buy", "taker_buy", "1", "true"):
            return 1
        if s in ("s", "sell", "taker_sell", "-1", "false"):
            return -1
        return 0
    df["side_num"] = df["side"].apply(side_to_num).astype(float)
    grouped = df.groupby(pd.Grouper(freq=resample_rule))
    agg = grouped.agg(
        ms_trade_count=("price", "count"),
        ms_vol=("size", "sum"),
        ms_avg_size=("size", "mean"),
        signed_vol=("side_num", lambda x: (x * df.loc[x.index, "size"]).sum()),
    )
    agg["ms_buy_vol"] = grouped.apply(lambda g: g.loc[g["side_num"] > 0, "size"].sum())
    agg["ms_sell_vol"] = grouped.apply(lambda g: g.loc[g["side_num"] < 0, "size"].sum())
    agg["ms_buy_sell_imb"] = (agg["ms_buy_vol"] - agg["ms_sell_vol"]) / agg["ms_vol"].replace(0, np.nan)
    agg["ms_cum_delta"] = agg["signed_vol"].cumsum()
    agg["ms_aggression_idx"] = (agg["signed_vol"] / agg["ms_vol"].replace(0, np.nan)).fillna(0).rolling(3, min_periods=1).mean()
    return agg

def depth_level_imbalance(depth_df: pd.DataFrame, levels: int = DEFAULT_LEVELS):
    if depth_df is None or depth_df.empty:
        return pd.DataFrame()
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    bids, asks = [], []
    for i in range(1, levels + 1):
        bcol = f"bid_{i}_size"
        acol = f"ask_{i}_size"
        if bcol in df.columns:
            bids.append(df[bcol].astype(float).fillna(0))
        else:
            bids.append(pd.Series(0.0, index=df.index))
        if acol in df.columns:
            asks.append(df[acol].astype(float).fillna(0))
        else:
            asks.append(pd.Series(0.0, index=df.index))
    bid_depth = pd.concat(bids, axis=1).sum(axis=1)
    ask_depth = pd.concat(asks, axis=1).sum(axis=1)
    out = {}
    for i, (b, a) in enumerate(zip(bids, asks), start=1):
        out[f"ms_imb_level_{i}"] = (b - a) / (b + a).replace(0, np.nan)
    out["ms_vol_imb_sum5"] = (bid_depth - ask_depth) / (bid_depth + ask_depth).replace(0, np.nan)
    return pd.DataFrame(out, index=df.index)

def book_pressure_delta(depth_df: pd.DataFrame, levels: int = DEFAULT_LEVELS):
    if depth_df is None or depth_df.empty:
        return pd.Series(np.nan, index=depth_df.index if depth_df is not None else [])
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    bid_v = []
    ask_v = []
    for i in range(1, levels + 1):
        bp = f"bid_{i}_price"
        bs = f"bid_{i}_size"
        ap = f"ask_{i}_price"
        asz = f"ask_{i}_size"
        if bp in df.columns and bs in df.columns:
            bid_v.append(df[bp].astype(float) * df[bs].astype(float))
        if ap in df.columns and asz in df.columns:
            ask_v.append(df[ap].astype(float) * df[asz].astype(float))
    if not bid_v and not ask_v:
        return pd.Series(np.nan, index=df.index, name=FEATURE_MANIFEST["book_pressure"])
    bid_sum = pd.concat(bid_v, axis=1).sum(axis=1) if bid_v else 0.0
    ask_sum = pd.concat(ask_v, axis=1).sum(axis=1) if ask_v else 0.0
    return ((bid_sum - ask_sum) / (bid_sum + ask_sum).replace(0, np.nan)).rename(FEATURE_MANIFEST["book_pressure"])

def top_book_pressure(depth_df: pd.DataFrame):
    """Top-level (level 1) pressure: (bid1_price*bid1_size - ask1_price*ask1_size)/(sum)"""
    if depth_df is None or depth_df.empty:
        return pd.Series(np.nan, index=depth_df.index if depth_df is not None else [], name=FEATURE_MANIFEST["top_book_pressure"])
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    if not all(k in df.columns for k in ("bid_1_price", "bid_1_size", "ask_1_price", "ask_1_size")):
        return pd.Series(np.nan, index=df.index, name=FEATURE_MANIFEST["top_book_pressure"])
    bp = (df["bid_1_price"].astype(float) * df["bid_1_size"].astype(float))
    ap = (df["ask_1_price"].astype(float) * df["ask_1_size"].astype(float))
    top = ((bp - ap) / (bp + ap).replace(0, np.nan)).rename(FEATURE_MANIFEST["top_book_pressure"])
    return top

def sweep_detector(trades_df: pd.DataFrame, price_col="price", size_col="size", side_col="side", window="1s", vol_thresh=3.0):
    if trades_df is None or trades_df.empty:
        return pd.Series(0, index=pd.DatetimeIndex([], dtype='datetime64[ns, UTC]'))
    df = trades_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    def sgn(x):
        s = str(x).lower()
        if s in ("b", "buy", "taker_buy"): return 1
        if s in ("s", "sell", "taker_sell"): return -1
        return 0
    df["snum"] = df[side_col].apply(sgn)
    df["signed_vol"] = df["snum"] * df[size_col].astype(float)
    agg = df["signed_vol"].resample(window).sum()
    z = (agg - agg.rolling(60, min_periods=1).mean()) / agg.rolling(60, min_periods=1).std().replace(0, np.nan)
    flag = (z.abs() > vol_thresh).astype(int)
    # return per-second (or window) flag indexed by the resampled index
    return flag

def compute_pull_stack_ratio(depth_df: pd.DataFrame, window="10s"):
    if depth_df is None or depth_df.empty:
        return pd.Series(0.0, index=depth_df.index if depth_df is not None else [])
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    if "bid_1_size" not in df.columns:
        return pd.Series(0.0, index=df.index)
    bid1 = df["bid_1_size"].astype(float).fillna(0)
    bid1_diff = bid1.diff().fillna(0)
    pull = (-bid1_diff).clip(lower=0).resample(window).sum()
    stack = bid1_diff.clip(lower=0).resample(window).sum()
    ratio = (pull / (stack.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0)
    return ratio.reindex(df.index, method="ffill").fillna(0).rename(FEATURE_MANIFEST["pull_stack_ratio"])

def compute_liq_decay(depth_df: pd.DataFrame, window="10s"):
    if depth_df is None or depth_df.empty or "bid_1_size" not in depth_df.columns:
        return pd.Series(0.0, index=depth_df.index if depth_df is not None else [])
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    top = df["bid_1_size"].astype(float).fillna(0)
    rolling_max = top.rolling(60, min_periods=1).max().replace(0, np.nan)
    decay = (rolling_max - top) / rolling_max
    return decay.fillna(0).rename(FEATURE_MANIFEST["liq_decay"])

def compute_spoof_absorption(trades_df: pd.DataFrame, depth_df: pd.DataFrame, window="10s"):
    if trades_df is None or trades_df.empty:
        spoof_rate = pd.Series(0.0, index=depth_df.index if depth_df is not None else [])
        absorption = pd.Series(0, index=depth_df.index if depth_df is not None else [])
        return spoof_rate.rename(FEATURE_MANIFEST["spoof_rate"]), absorption.rename(FEATURE_MANIFEST["absorption_flag"])
    t = trades_df.copy()
    if "datetime" in t.columns:
        t["datetime"] = pd.to_datetime(t["datetime"], utc=True)
        t = t.set_index("datetime").sort_index()
    # absorption heuristic: large trades relative to rolling mean
    signed_size = t.get("size", t.index.to_series()*0).astype(float)
    avg_size = signed_size.rolling("1min").mean().reindex(t.index, method="ffill").fillna(signed_size.mean() if len(signed_size) else 1.0)
    large = (signed_size > 5 * (avg_size.replace(0, 1.0))).astype(int)
    absorption = large.resample(window).max()
    absorption = absorption.reindex(depth_df.index if depth_df is not None else t.index, method="ffill").fillna(0).astype(int)
    # spoof rate placeholder (requires cancels) -> 0 default
    spoof_rate = pd.Series(0.0, index=depth_df.index if depth_df is not None else t.index)
    return spoof_rate.rename(FEATURE_MANIFEST["spoof_rate"]), absorption.rename(FEATURE_MANIFEST["absorption_flag"])

def compute_orderflow_vel(trades_df: pd.DataFrame, resample_rule: str = "1min"):
    """Orderflow velocity: change in signed volume per second (resampled)."""
    if trades_df is None or trades_df.empty:
        return pd.Series(0.0, index=pd.DatetimeIndex([], dtype='datetime64[ns, UTC]'))
    df = trades_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    def sgn(x):
        s = str(x).lower()
        if s in ("b", "buy", "taker_buy"): return 1
        if s in ("s", "sell", "taker_sell"): return -1
        return 0
    df["snum"] = df.get("side_num", df.get("side", df["size"])).apply(lambda x: sgn(x) if isinstance(x, str) else (1 if x > 0 else (-1 if x < 0 else 0)))
    df["signed_notional"] = df["snum"] * df.get("size", df.get("notional", df.get("size", 0))).astype(float)
    agg = df["signed_notional"].resample(resample_rule).sum().fillna(0)
    # compute velocity as difference / seconds in resample window
    # determine seconds for resample_rule (basic heuristics)
    if "min" in resample_rule or "T" in resample_rule:
        sec = 60 * int(''.join(filter(str.isdigit, resample_rule)) or 1)
    elif "s" in resample_rule.lower():
        sec = int(''.join(filter(str.isdigit, resample_rule)) or 1)
    else:
        sec = 60
    vel = agg.diff().fillna(0) / max(sec, 1)
    return vel.rename(FEATURE_MANIFEST["orderflow_vel"])

def compute_heat_slope(depth_df: pd.DataFrame, levels: int = DEFAULT_LEVELS):
    """
    Compute slope of imbalance across book levels:
    For each timestamp take vector v = (bid1_size - ask1_size, bid2_size - ask2_size, ...)
    compute linear slope across level index.
    """
    if depth_df is None or depth_df.empty:
        return pd.Series(0.0, index=depth_df.index if depth_df is not None else [])
    df = depth_df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    # build matrix of imbalances
    mats = []
    for i in range(1, levels + 1):
        bcol = f"bid_{i}_size"
        acol = f"ask_{i}_size"
        b = df[bcol].astype(float) if bcol in df.columns else pd.Series(0.0, index=df.index)
        a = df[acol].astype(float) if acol in df.columns else pd.Series(0.0, index=df.index)
        mats.append((b - a).fillna(0.0))
    if not mats:
        return pd.Series(0.0, index=df.index)
    mat = pd.concat(mats, axis=1).fillna(0.0).values  # shape (T, L)
    # compute slope across level axis for each row
    idxs = np.arange(1, mat.shape[1] + 1)
    slopes = []
    for row in mat:
        if np.all(row == 0):
            slopes.append(0.0)
            continue
        # robust linear fit
        try:
            coef = np.polyfit(idxs, row, 1)[0]
            slopes.append(float(coef))
        except Exception:
            slopes.append(0.0)
    return pd.Series(slopes, index=df.index).rename(FEATURE_MANIFEST["heat_slope"])

def micro_divergence(bar_df: pd.DataFrame, trade_agg_df: pd.DataFrame):
    if trade_agg_df is None or trade_agg_df.empty:
        return pd.Series(np.nan, index=bar_df.index, name=FEATURE_MANIFEST["micro_div"])
    t = trade_agg_df.reindex(bar_df.index, method="ffill").fillna(0)
    cvd = t["ms_cum_delta"].astype(float).fillna(0)
    price_ret = bar_df["close"].astype(float).pct_change().fillna(0)
    cvd_ret = cvd.diff().fillna(0)
    score = (price_ret.rolling(5).mean() - cvd_ret.rolling(5).mean()).fillna(0)
    return score.rename(FEATURE_MANIFEST["micro_div"])

def compute_microstructure_features(bar_df: pd.DataFrame,
                                     trades_df: Optional[pd.DataFrame] = None,
                                     depth_df: Optional[pd.DataFrame] = None,
                                     params: Optional[Dict] = None) -> pd.DataFrame:
    params = params or {}
    bar_df = _ensure_indexed(bar_df)
    resample_rule = params.get("resample_rule", "1min")
    levels = int(params.get("levels", DEFAULT_LEVELS))

    parts = []
    # core spread/mid
    parts.append(_spread_mid(bar_df))

    # trade aggregated features
    trade_agg = trade_agg_features(trades_df, resample_rule=resample_rule) if trades_df is not None else pd.DataFrame(index=bar_df.index)
    if not trade_agg.empty:
        trade_agg = trade_agg.reindex(bar_df.index, method="ffill").fillna(0)
        parts.append(trade_agg)

    # depth-derived
    if depth_df is not None:
        dlev = depth_level_imbalance(depth_df, levels=levels)
        if not dlev.empty:
            dlev = dlev.reindex(bar_df.index, method="ffill").fillna(0)
            parts.append(dlev)
        bp = book_pressure_delta(depth_df, levels=levels)
        if isinstance(bp, pd.Series):
            bp = bp.reindex(bar_df.index, method="ffill").fillna(0)
            parts.append(bp.to_frame())
        # top book pressure
        try:
            tbp = top_book_pressure(depth_df).reindex(bar_df.index, method="ffill").fillna(0)
            parts.append(tbp.to_frame())
        except Exception:
            parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["top_book_pressure"]).to_frame())

    # sweep (from trades) -> resampled to bars
    try:
        sw = sweep_detector(trades_df, window=params.get("sweep_window", "1s"), vol_thresh=float(params.get("sweep_vol_thresh", 3.0)))
        if isinstance(sw.index, pd.DatetimeIndex) and len(sw):
            sw_bar = sw.resample(resample_rule).max().reindex(bar_df.index, method="ffill").fillna(0).astype(int)
        else:
            sw_bar = pd.Series(0, index=bar_df.index)
        parts.append(sw_bar.rename(FEATURE_MANIFEST["sweep_flag"]).to_frame())
    except Exception:
        parts.append(pd.Series(0, index=bar_df.index, name=FEATURE_MANIFEST["sweep_flag"]).to_frame())

    # pull/stack ratio and liq decay from depth
    try:
        if depth_df is not None:
            ps = compute_pull_stack_ratio(depth_df, window=params.get("pull_stack_window", "10s"))
            parts.append(ps.reindex(bar_df.index, method="ffill").fillna(0))
            ld = compute_liq_decay(depth_df, window=params.get("liq_decay_window", "10s"))
            parts.append(ld.reindex(bar_df.index, method="ffill").fillna(0))
        else:
            parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["pull_stack_ratio"]))
            parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["liq_decay"]))
    except Exception:
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["pull_stack_ratio"]))
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["liq_decay"]))

    # spoof / absorption heuristics
    try:
        sp, ab = compute_spoof_absorption(trades_df, depth_df, window=params.get("spoof_window", "10s"))
        parts.append(sp.reindex(bar_df.index, method="ffill").fillna(0))
        parts.append(ab.reindex(bar_df.index, method="ffill").fillna(0))
    except Exception:
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["spoof_rate"]))
        parts.append(pd.Series(0, index=bar_df.index, name=FEATURE_MANIFEST["absorption_flag"]))

    # orderflow velocity (resampled)
    try:
        ov = compute_orderflow_vel(trades_df, resample_rule=resample_rule)
        parts.append(ov.reindex(bar_df.index, method="ffill").fillna(0))
    except Exception:
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["orderflow_vel"]))

    # microtrend vector: short-term sign of buy/sell imbalance momentum
    try:
        if not trade_agg.empty:
            imb = trade_agg.reindex(bar_df.index, method="ffill").fillna(0)["ms_buy_sell_imb"]
            microtrend = imb.rolling(int(params.get("microtrend_win", 3)), min_periods=1).mean().apply(np.sign).fillna(0)
            parts.append(microtrend.rename(FEATURE_MANIFEST["microtrend_vector"]))
        else:
            parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["microtrend_vector"]))
    except Exception:
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["microtrend_vector"]))

    # heatmap slope across levels
    try:
        if depth_df is not None:
            hs = compute_heat_slope(depth_df, levels=levels)
            parts.append(hs.reindex(bar_df.index, method="ffill").fillna(0))
        else:
            parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["heat_slope"]))
    except Exception:
        parts.append(pd.Series(0.0, index=bar_df.index, name=FEATURE_MANIFEST["heat_slope"]))

    # add micro divergence if trade agg present
    if not trade_agg.empty:
        parts.append(micro_divergence(bar_df, trade_agg).to_frame())

    # final assembly
    out = pd.concat([p if isinstance(p, pd.DataFrame) else p.to_frame() for p in parts], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    print("[ms] computed features:", list(out.columns)[:200])
    if not out.empty:
        print("[ms] non-null sample counts:", {c: int(out[c].notna().sum()) for c in out.columns[:10]})
    return out
