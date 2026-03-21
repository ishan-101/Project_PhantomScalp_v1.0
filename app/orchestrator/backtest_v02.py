# app/orchestrator/backtest_v02.py
from __future__ import annotations
"""
backtest_v02.py (patched)

Wires DataIO -> features -> labelers -> policy -> router -> simple fills -> PnL -> report bundle.

Policy defaults:
  - require regime==1 for CALL_LONG, regime==2 for PUT_LONG
  - reversal must be 0
  - cooldown default: 15 minutes (configurable)
  - hold_minutes default: 3 (configurable)
  - SL = 0.5 * ATR (atr length 14), TP = 1.0 * ATR, trail_step = 0.75 * ATR
"""

# --------------------------------------------------
# Ensure project root is added to sys.path 
# so all "app.*" imports work when running file directly
# --------------------------------------------------
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root (two levels up)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------
# Standard / third-party imports
# -----------------------------
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import math

# -----------------------------
# Plotly lazy import
# -----------------------------
def _import_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore
        return go
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Plotly is required for charts. Install with:\n"
            "  python -m pip install plotly kaleido"
        ) from e

# -----------------------------
# Report bundle (resilient import path)
# -----------------------------
try:
    from app.analytics.report_bundle import ReportBundle  # preferred
except Exception:
    from analytics.report_bundle import ReportBundle      # fallback if flat

# Optional: BacktestConfig (not required)
try:
    from app.schemas import BacktestConfig  # noqa
except Exception:
    BacktestConfig = None  # type: ignore

# -----------------------------
# Data / feature / label imports
# -----------------------------
try:
    from app.dataio import DataBundle
except Exception:
    from dataio import DataBundle  # type: ignore

try:
    from app.features.time_cycle import compute_time_cycle_features, atr as tc_atr_helper
    from app.features.microstructure import compute_microstructure_features
    from app.features.options_features import compute_options_features
except Exception:
    from features.time_cycle import compute_time_cycle_features, atr as tc_atr_helper  # type: ignore
    from features.microstructure import compute_microstructure_features  # type: ignore
    from features.options_features import compute_options_features  # type: ignore

# labelers
try:
    from app.ml.labels.regime import label_regime
    from app.ml.labels.reversal import label_reversal
    from app.ml.labels.cycle import label_cycle
except Exception:
    from ml.labels.regime import label_regime  # type: ignore
    from ml.labels.reversal import label_reversal  # type: ignore
    from ml.labels.cycle import label_cycle  # type: ignore

# -----------------------------
# Helpers (KPI / equity)
# -----------------------------
def _to_dict(cfg: Any) -> Dict[str, Any]:
    try:
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        if hasattr(cfg, "dict"):
            return cfg.dict()
        if is_dataclass(cfg):
            return asdict(cfg)
        if hasattr(cfg, "__dict__"):
            return dict(cfg.__dict__)
    except Exception:
        pass
    return {"cfg": str(cfg)}

def _equity_drawdown(trades: pd.DataFrame, timeline: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    if trades.empty:
        eq = pd.Series(0.0, index=timeline, name="equity")
        dd = pd.Series(0.0, index=timeline, name="drawdown")
        return eq, dd

    equity_ts = (
        trades.assign(equity=trades["pnl"].cumsum())
              .set_index(pd.to_datetime(trades["time_out"]))["equity"]
              .sort_index()
    )
    equity = equity_ts.reindex(timeline, method="ffill").fillna(0.0)
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0.0, np.nan)
    dd = dd.fillna(0.0)
    equity.name = "equity"
    dd.name = "drawdown"
    return equity, dd

def _kpis_from(trades: pd.DataFrame, equity_shifted: pd.Series, dd: pd.Series, starting_capital: float) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}
    n_trades = int(len(trades))
    kpis["n_trades"] = n_trades
    kpis["starting_capital"] = float(starting_capital)

    if n_trades == 0 or equity_shifted.empty:
        kpis.update({
            "ending_capital": float(starting_capital),
            "net_profit": 0.0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "max_dd": float(dd.min()) if len(dd) else 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "avg_rr": 0.0,
        })
        return kpis

    ending_capital = float(equity_shifted.iloc[-1])
    kpis["ending_capital"] = ending_capital
    kpis["net_profit"] = ending_capital - starting_capital
    kpis["total_return"] = (ending_capital - starting_capital) / max(starting_capital, 1e-12)

    pnl = trades["pnl"].astype(float)
    kpis["hit_rate"] = float((pnl > 0).mean())
    kpis["max_dd"] = float(dd.min())

    rets = equity_shifted.diff().fillna(0.0)
    std = rets.std()
    mean = rets.mean()
    if std == 0:
        kpis["sharpe"] = 0.0
        kpis["sortino"] = 0.0
    else:
        ann = np.sqrt(1440.0)
        downside = rets[rets < 0].std() or 1e-12
        kpis["sharpe"] = float((mean / std) * ann)
        kpis["sortino"] = float((mean / downside) * ann)

    wins = trades.loc[trades["pnl"] > 0, "pnl"]
    losses = -trades.loc[trades["pnl"] < 0, "pnl"]
    if len(wins) and len(losses):
        kpis["avg_rr"] = float(wins.mean() / max(losses.mean(), 1e-12))
    else:
        kpis["avg_rr"] = 0.0

    return kpis

def _rolling_winrate(trades: pd.DataFrame, window: str = "1D") -> Optional[pd.Series]:
    if trades.empty:
        return None
    t_sorted = trades.sort_values("time_out").copy()
    t_sorted["is_win"] = (t_sorted["pnl"] > 0).astype(float)
    roll = (
        t_sorted.set_index(pd.to_datetime(t_sorted["time_out"]))["is_win"]
                .rolling(window).mean().dropna()
    )
    return roll

# -----------------------------
# Main orchestrator
# -----------------------------
def run_backtest_v02(cfg) -> Dict[str, Any]:
    """
    cfg expected: output_dir, mode, symbol, start, end, seed,
                  hold_minutes (default 3), cooldown_min (default 15),
                  initial_capital
    """
    go = _import_plotly()
    np.random.seed(getattr(cfg, "seed", 42))

    # Data IO
    mode = getattr(cfg, "mode", "file")
    symbol = getattr(cfg, "symbol", "BTCUSDT")
    start = getattr(cfg, "start", None)
    end = getattr(cfg, "end", None)
    bundle = DataBundle(mode=mode, fixtures_dir=getattr(cfg, "fixtures_dir", "data/fixtures"))

    print(f"[v0.2] Starting backtest | mode={mode} symbol={symbol} window=({start} → {end})")
    if mode == "live":
        print("[DataIO] Live mode: fetching from APIs...")

    print("[DataIO] Fetching spot OHLCV...")
    spot_df = bundle.load_spot_ohlcv(symbol=symbol, interval="1m", start=start, end=end)
    print(f"[DataIO] Spot rows: {len(spot_df)}")
    if spot_df.empty:
        raise RuntimeError("No spot data returned. Check dates/network/rate-limit.")

    # normalize index to timezone-aware UTC datetimes
    spot_df = spot_df.copy()
    if "datetime" in spot_df.columns:
        spot_df["datetime"] = pd.to_datetime(spot_df["datetime"], utc=True)
        spot_df = spot_df.set_index("datetime").sort_index()
    else:
        spot_df.index = pd.to_datetime(spot_df.index, utc=True)
    ts = spot_df.index

    # Features
    print("[Features] Computing time-cycle features...")
    tc = compute_time_cycle_features(spot_df)
    print(f"[Features] TC cols: {len(tc.columns)}")

    print("[Features] Computing microstructure features (spot only)...")
    trades_df = None
    if hasattr(bundle, "load_spot_trades"):
        try:
            trades_df = bundle.load_spot_trades(symbol=symbol, start=start, end=end)
        except Exception:
            trades_df = None
    ms = compute_microstructure_features(spot_df, trades_df=trades_df)
    print(f"[Features] MS cols: {len(ms.columns)}")

    opt_df = None
    try:
        if hasattr(bundle, "load_option_ohlc"):
            opt_df = bundle.load_option_ohlc(symbol=symbol, start=start, end=end)
    except Exception:
        opt_df = None

    if opt_df is not None:
        print("[Features] Computing options features...")
        optf = compute_options_features(opt_df, spot_series=spot_df["close"])
        print(f"[Features] OPT cols: {len(optf.columns)}")
    else:
        optf = None

    # Labels
    print("[Labels] Building regime & reversal labels...")
    regime = label_regime(spot_df)
    reversal = label_reversal(spot_df, features_df=ms)
    cycle = label_cycle(spot_df)

    signals = pd.DataFrame(index=ts)
    signals["regime"] = regime.reindex(ts, method="ffill").fillna(0).astype(int)
    signals["reversal"] = reversal.reindex(ts, method="ffill").fillna(0).astype(int)
    signals["cycle"] = cycle.reindex(ts, method="ffill").fillna(0).astype(int)

    def _policy_action(rg:int, rev:int) -> str:
        if rg == 1 and rev == 0:
            return "CALL_LONG"
        if rg == 2 and rev == 0:
            return "PUT_LONG"
        return "FLAT"

    signals["action"] = [ _policy_action(int(rg), int(rv)) for rg, rv in zip(signals["regime"], signals["reversal"]) ]

    # Router/execution policy params
    cooldown_min = int(getattr(cfg, "cooldown_min", 15))
    hold_minutes = int(getattr(cfg, "hold_minutes", 3))
    sl_atr_mult = float(getattr(cfg, "sl_atr_mult", 0.5))
    tp_atr_mult = float(getattr(cfg, "tp_atr_mult", 1.0))
    trail_atr_mult = float(getattr(cfg, "trail_atr_mult", 0.75))

    print(f"[Router] Entries (policy applied): {int((signals['action'] != 'FLAT').sum())} | Hold≤{hold_minutes}m | SL={sl_atr_mult}×ATR TP={tp_atr_mult}×ATR Trail={trail_atr_mult}×ATR")

    candidate_ts = signals.index[signals["action"].isin(["CALL_LONG", "PUT_LONG"])]

    selected_entries = []
    last_entry_time = pd.Timestamp.min.tz_localize(ts.tz) if getattr(ts, "tz", None) is not None else pd.Timestamp.min
    for t in candidate_ts:
        if (t - last_entry_time) < pd.Timedelta(minutes=cooldown_min):
            continue
        selected_entries.append(t)
        last_entry_time = t

    print(f"[Router] Entries (after cooldown): {len(selected_entries)}")

    trades_rec = []
    for tin in selected_entries:
        side = signals.at[tin, "action"]
        px_in = float(spot_df["close"].at[tin])

        # ATR at entry
        atr_val = None
        if isinstance(tc, pd.DataFrame) and "tc_atr" in tc.columns:
            try:
                atr_val = float(tc["tc_atr"].reindex([tin], method="ffill").iloc[0])
            except Exception:
                atr_val = None
        if atr_val is None or math.isnan(atr_val):
            atr_val = float(spot_df["close"].pct_change().rolling(14, min_periods=1).std().iloc[-1] * px_in)

        sl = sl_atr_mult * atr_val
        tp = tp_atr_mult * atr_val
        trail_step = trail_atr_mult * atr_val

        tout_time = tin + pd.Timedelta(minutes=hold_minutes)
        if tout_time > ts[-1]:
            continue

        window_slice = spot_df.loc[tin:tout_time]["close"].sort_index()

        direction = 1 if side == "CALL_LONG" else -1
        entry_price = px_in

        stop_price = entry_price - direction * sl
        take_price = entry_price + direction * tp
        best_price = entry_price

        exited = False
        exit_time = tout_time
        exit_price = float(window_slice.iloc[-1])

        # iterate minute bars after entry
        for t_idx, p in window_slice.iloc[1:].items():
            p = float(p)
            if direction == 1:
                if p > best_price:
                    best_price = p
                if (best_price - p) >= trail_step:
                    exit_time = t_idx; exit_price = p; exited = True; break
                if p <= stop_price:
                    exit_time = t_idx; exit_price = p; exited = True; break
                if p >= take_price:
                    exit_time = t_idx; exit_price = p; exited = True; break
            else:
                if p < best_price:
                    best_price = p
                if (p - best_price) >= trail_step:
                    exit_time = t_idx; exit_price = p; exited = True; break
                if p >= stop_price:
                    exit_time = t_idx; exit_price = p; exited = True; break
                if p <= take_price:
                    exit_time = t_idx; exit_price = p; exited = True; break

        pnl = (exit_price - entry_price) * direction
        ret = pnl / max(entry_price, 1e-12)
        trades_rec.append({
            "time_in": tin.isoformat(),
            "time_out": exit_time.isoformat(),
            "side": side,
            "qty": 1.0,
            "entry": float(entry_price),
            "exit": float(exit_price),
            "pnl": float(pnl),
            "ret": float(ret),
            "win": bool(pnl > 0.0),
        })

    trades = pd.DataFrame(trades_rec, columns=["time_in", "time_out", "side", "qty", "entry", "exit", "pnl", "ret", "win"])

    # Equity/KPIs
    starting_capital = float(getattr(cfg, "initial_capital", 1000.0))
    base_equity, dd = _equity_drawdown(trades, ts)
    equity = base_equity + starting_capital
    kpis = _kpis_from(trades, equity, dd, starting_capital)

    # Report bundle
    outdir = Path(getattr(cfg, "output_dir", "./out/v02"))
    outdir.mkdir(parents=True, exist_ok=True)
    meta_cfg = _to_dict(cfg)
    meta_cfg.update({"currency": "INR"})

    bundle_report = (ReportBundle(outdir, title="PhantomScalp v0.2 Backtest (INR)")
                     .add_meta(config=meta_cfg, kpis=kpis))

    # Charts
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity (INR)"))
    eq_fig.update_layout(title="Equity Curve (INR)", xaxis_title="Time", yaxis_title="Equity (₹)")
    bundle_report.add_figure("equity_curve", eq_fig, "Cumulative equity over time (starting ₹%g)." % starting_capital)

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100.0, mode="lines", name="Drawdown %"))
    dd_fig.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown (%)")
    bundle_report.add_figure("drawdown", dd_fig, "Underwater curve (percentage from peak).")

    roll = _rolling_winrate(trades, "1D")
    if roll is not None and len(roll):
        wr_fig = go.Figure()
        wr_fig.add_trace(go.Scatter(x=roll.index, y=roll.values * 100.0, mode="lines", name="Win% (1D)"))
        wr_fig.update_layout(title="Rolling Win Rate (1D)", xaxis_title="Time", yaxis_title="Win Rate (%)")
        bundle_report.add_figure("rolling_winrate", wr_fig, "Rolling daily win percentage.")

    # Tables
    bundle_report.add_table("trades", trades, "All executed trades.", as_csv=True)
    bundle_report.add_table("equity", pd.DataFrame({"ts": equity.index, "equity_inr": equity.values}), "Equity time series (INR).", as_csv=True)
    bundle_report.add_table("signals", signals.reset_index().rename(columns={"index": "ts"}), "Model signals + policy actions.", as_csv=True)

    paths = bundle_report.save()

    flat_paths = {"summary_json": paths.get("summary_json"), "report_html": paths.get("report_html")}
    for k, v in list(paths.items()):
        if k.startswith("csv::") or k.startswith("img::"):
            flat_paths[k] = v

    print("[Done] Artifacts saved under:", outdir)
    return {"paths": flat_paths, "kpis": kpis}

# -----------------------------
# CLI smoke test
# -----------------------------
if __name__ == "__main__":
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        output_dir="./out/v02_demo",
        seed=123,
        mode="file",
        symbol="BTCUSDT",
        start="2024-01-01",
        end="2024-01-02",
        hold_minutes=3,
        cooldown_min=15,
        initial_capital=1000.0
    )
    res = run_backtest_v02(cfg)
    print("Artifacts:")
    for k, v in res["paths"].items():
        print(f"  {k}: {v}")
    print("KPIs:", res["kpis"])
