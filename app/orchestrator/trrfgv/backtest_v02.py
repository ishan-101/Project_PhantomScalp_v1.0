# app/orchestrator/backtest_v02.py
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Plotly (lazy import)
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
    from app.analytics.report_bundle import ReportBundle
except Exception:
    from analytics.report_bundle import ReportBundle

try:
    from app.schemas import BacktestConfig  # noqa
except Exception:
    BacktestConfig = None  # type: ignore

# -----------------------------
# DataIO bundle
# -----------------------------
try:
    from app.dataio import DataBundle
    from app.dataio.binance_spot import BinanceSpotClient
    from app.dataio.delta_options import DeltaOptionsClient
except Exception:
    from dataio import DataBundle
    from dataio.binance_spot import BinanceSpotClient
    from dataio.delta_options import DeltaOptionsClient


# -----------------------------
# Helpers
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


def _kpis_from(
    trades: pd.DataFrame,
    equity_shifted: pd.Series,
    dd: pd.Series,
    starting_capital: float
) -> Dict[str, Any]:
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
        ann = np.sqrt(1440.0)  # rough per-minute -> daily
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
# Orchestrator
# -----------------------------
def run_backtest_v02(cfg) -> Dict[str, Any]:
    """
    v0.2 backtest orchestrator:
      data -> features -> labels -> model inference -> policy -> router -> fills -> PnL -> report
    """
    go = _import_plotly()
    np.random.seed(getattr(cfg, "seed", 42))

    # 1) DATA (via DataIO bundle)
    data_mode = getattr(cfg, "data_mode", "live")
    fixtures_dir = Path(getattr(cfg, "fixtures_dir", "data/fixtures"))
    symbol = getattr(cfg, "symbol", "BTCUSDT")
    start = getattr(cfg, "start", dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))
    end   = getattr(cfg, "end",   dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc))  # default 1 day for speed

    print(f"[v0.2] Starting backtest | mode={data_mode} symbol={symbol} window=({start.isoformat()} → {end.isoformat()})")

    if data_mode == "file":
        spot_fixture = fixtures_dir / "binance_btcusdt_1m.csv"
        spot = BinanceSpotClient(mode="file", file_path=str(spot_fixture))
        opt = DeltaOptionsClient(mode="file", file_map={})
        bundle = DataBundle(spot_client=spot, options_client=opt)
        print(f"[DataIO] File mode: {spot_fixture}")
    else:
        bundle = DataBundle()
        print("[DataIO] Live mode: fetching from APIs...")

    print("[DataIO] Fetching spot OHLCV...")
    spot_df = bundle.load_spot_ohlcv(symbol=symbol, interval="1m", start=start, end=end)
    print(f"[DataIO] Spot rows: {len(spot_df)}")

    if spot_df.empty:
        raise RuntimeError("No spot data returned. Check dates/network/rate-limit.")

    ts = pd.to_datetime(spot_df["datetime"])
    price = spot_df.set_index("datetime")["close"]

    # === ATR(14) on spot for SL/TP/trailing ===
    hl = spot_df["high"] - spot_df["low"]
    hc = (spot_df["high"] - spot_df["close"].shift()).abs()
    lc = (spot_df["low"] - spot_df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()

    # 2) SIGNALS (stubbed for now)
    print("[Signals] Generating stub signals...")
    signals = pd.DataFrame({
        "ts": ts,
        "regime_sig": np.random.choice([0, 1, 2], size=len(ts), p=[0.4, 0.3, 0.3]),
        "reversal_sig": np.random.choice([0, 1], size=len(ts), p=[0.85, 0.15]),
        "cycle_sig": np.random.choice([0, 1, 2], size=len(ts)),
    }).set_index("ts")

    # 3) ROUTER with entry filter + cooldown + ATR exits
    # Entry filter: strong conditions only
    mask = (
        ((signals["regime_sig"] == 1) & (signals["reversal_sig"] == 0)) |  # CALL bias
        ((signals["regime_sig"] == 2) & (signals["reversal_sig"] == 0))    # PUT bias
    )
    candidate_entries = signals[mask].index

    # Cooldown (default 20m), configurable
    cooldown = pd.Timedelta(minutes=int(getattr(cfg, "cooldown_minutes", 20)))
    entries = []
    last = None
    for t in candidate_entries:
        if last is None or t - last >= cooldown:
            entries.append(t)
            last = t
    entries = pd.DatetimeIndex(entries)

    # Exit params (defaults): hold<=3m, SL=0.5*ATR, TP=1.0*ATR, trail=0.75*ATR
    max_hold = pd.Timedelta(minutes=int(getattr(cfg, "hold_minutes", 3)))
    sl_mult = float(getattr(cfg, "sl_atr_mult", 0.5))
    tp_mult = float(getattr(cfg, "tp_atr_mult", 1.0))
    trail_mult = float(getattr(cfg, "trail_atr_mult", 0.75))

    print(f"[Router] Entries (after cooldown): {len(entries)} | Hold≤{int(max_hold.total_seconds()/60)}m | SL={sl_mult}×ATR TP={tp_mult}×ATR Trail={trail_mult}×ATR")

    trades_rec = []
    for tin in entries:
        # Map regime to side; (1 -> CALL), (2 -> PUT)
        side = "CALL_LONG" if signals.loc[tin, "regime_sig"] == 1 else "PUT_LONG"
        direction = 1 if side == "CALL_LONG" else -1
        px_in = float(price.loc[tin])

        atr_val = float(atr14.loc[tin]) if tin in atr14.index and pd.notna(atr14.loc[tin]) else None
        if not atr_val or not np.isfinite(atr_val):
            continue  # skip until ATR is ready (first 14 bars)

        # Static SL/TP levels
        sl_level = px_in - sl_mult * atr_val * direction
        tp_level = px_in + tp_mult * atr_val * direction
        # Initialize trailing in adverse/protective direction
        trail_level = px_in - trail_mult * atr_val * direction

        tout_limit = tin + max_hold
        px_out = px_in
        tout = tout_limit

        # Step forward until SL/TP/trail hit or time limit
        # Ensure we only iterate over available bars
        series = price.loc[tin:tout_limit]
        if len(series) > 1:
            for t, px in series.iloc[1:].items():
                # evolve trailing stop with favorable move
                if direction == 1:
                    trail_level = max(trail_level, px - trail_mult * atr_val)
                    hit_sl = (px <= sl_level) or (px <= trail_level)
                    hit_tp = (px >= tp_level)
                else:
                    trail_level = min(trail_level, px + trail_mult * atr_val)
                    hit_sl = (px >= sl_level) or (px >= trail_level)
                    hit_tp = (px <= tp_level)

                if hit_sl or hit_tp:
                    px_out = float(px)
                    tout = t
                    break
                px_out = float(px)  # last seen price (for time-based exit)

        pnl = (px_out - px_in) * direction
        ret = pnl / max(px_in, 1e-12)
        trades_rec.append({
            "time_in": tin.isoformat(),
            "time_out": tout.isoformat(),
            "side": side,
            "qty": 1.0,
            "entry": px_in,
            "exit": px_out,
            "pnl": float(pnl),
            "ret": float(ret),
            "win": bool(pnl > 0.0),
        })

    trades = pd.DataFrame(trades_rec)
    print(f"[Router] Trades generated: {len(trades)}")

    # 4) EQUITY / DRAWDOWN / KPIs
    starting_capital = float(getattr(cfg, "initial_capital", 1000.0))
    base_equity, dd = _equity_drawdown(trades, ts)
    equity = base_equity + starting_capital
    kpis = _kpis_from(trades, equity, dd, starting_capital)

    # 5) REPORT BUNDLE
    outdir = Path(getattr(cfg, "output_dir", "./out/v02"))
    outdir.mkdir(parents=True, exist_ok=True)
    meta_cfg = _to_dict(cfg)
    meta_cfg.update({"currency": "INR"})

    bundle_out = (ReportBundle(outdir, title="PhantomScalp v0.2 Backtest (INR)")
                  .add_meta(config=meta_cfg, kpis=kpis))

    go = _import_plotly()
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity (INR)"))
    eq_fig.update_layout(title="Equity Curve (INR)", xaxis_title="Time", yaxis_title="Equity (₹)")
    bundle_out.add_figure("equity_curve", eq_fig, "Cumulative equity over time.")

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100.0, mode="lines", name="Drawdown %"))
    dd_fig.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown (%)")
    bundle_out.add_figure("drawdown", dd_fig, "Underwater curve.")

    roll = _rolling_winrate(trades, "1D")
    if roll is not None and len(roll):
        wr_fig = go.Figure()
        wr_fig.add_trace(go.Scatter(x=roll.index, y=roll.values * 100.0, mode="lines", name="Win% (1D)"))
        wr_fig.update_layout(title="Rolling Win Rate (1D)", xaxis_title="Time", yaxis_title="Win Rate (%)")
        bundle_out.add_figure("rolling_winrate", wr_fig, "Rolling daily win percentage.")

    # Tables
    bundle_out.add_table("trades", trades, "All executed trades.", as_csv=True)
    bundle_out.add_table(
        "equity",
        pd.DataFrame({"ts": equity.index, "equity_inr": equity.values}),
        "Equity time series (INR).",
        as_csv=True
    )
    bundle_out.add_table(
        "signals",
        signals.reset_index(),
        "Model signals + policy actions.",
        as_csv=True
    )

    paths = bundle_out.save()

    flat_paths = {"summary_json": paths.get("summary_json"), "report_html": paths.get("report_html")}
    for k, v in list(paths.items()):
        if k and (k.startswith("csv::") or k.startswith("img::")):
            flat_paths[k] = v

    print(f"[Done] Artifacts saved under: {outdir}")
    return {"paths": flat_paths, "kpis": kpis}


# -----------------------------
# CLI
# -----------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="PhantomScalp v0.2 backtest")
    p.add_argument("--output-dir", default="./out/v02_demo")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--hold-minutes", type=int, default=3, help="Max hold in minutes (default 3)")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default=None, help="ISO datetime, e.g. 2024-01-01T00:00:00Z")
    p.add_argument("--end", default=None, help="ISO datetime, e.g. 2024-01-02T00:00:00Z")
    p.add_argument("--data-mode", choices=["live", "file"], default="live")
    p.add_argument("--fixtures-dir", default="data/fixtures")
    p.add_argument("--cooldown-minutes", type=int, default=20)
    p.add_argument("--sl-atr-mult", type=float, default=0.5)
    p.add_argument("--tp-atr-mult", type=float, default=1.0)
    p.add_argument("--trail-atr-mult", type=float, default=0.75)
    return p.parse_args()


def _parse_iso(ts: Optional[str], default: dt.datetime) -> dt.datetime:
    if not ts:
        return default
    ts = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    d = dt.datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


if __name__ == "__main__":
    args = _parse_args()
    start = _parse_iso(args.start, dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))
    end = _parse_iso(args.end, dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc))

    from types import SimpleNamespace
    cfg = SimpleNamespace(
        output_dir=args.output_dir,
        seed=args.seed,
        hold_minutes=args.hold_minutes,
        initial_capital=args.initial_capital,
        symbol=args.symbol,
        start=start,
        end=end,
        data_mode=args.data_mode,
        fixtures_dir=args.fixtures_dir,
        cooldown_minutes=args.cooldown_minutes,
        sl_atr_mult=args.sl_atr_mult,
        tp_atr_mult=args.tp_atr_mult,
        trail_atr_mult=args.trail_atr_mult,
    )

    res = run_backtest_v02(cfg)
    print("Artifacts:")
    for k, v in res["paths"].items():
        print(f"  {k}: {v}")
    print("KPIs:", res["kpis"])
