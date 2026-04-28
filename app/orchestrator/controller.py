# app/orchestrator/controller.py
from __future__ import annotations

"""
Project_PhantomScalp v1.0 (v0.2)
Controller module:
- Preserves v0.1-style backtest API: Orchestrator.run_backtest()
- Re-exports v0.2 pipeline implemented in app/orchestrator/backtest_v02.py
  via: from .backtest_v02 import run_backtest_v02
- Provides an instance wrapper Orchestrator.run_backtest_v02(...) that delegates
  to the function above, so callers can choose functional or OOP styles.

This keeps controller.py slim and future-proof.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Resilient imports for v0.1/v0.2 naming mismatches
# -----------------------------------------------------------------------------

# config
try:
    from app.common.config import load_settings
except Exception:
    try:
        from app.common.config import get_settings as load_settings  # fallback alias
    except Exception as _e:
        raise ImportError("Could not import load_settings/get_settings from app.common.config") from _e

# logging
try:
    from app.telemetry.logs import get_logger
except Exception:
    def get_logger(level: str = "INFO"):
        import logging, sys
        lg = logging.getLogger("phantom")
        if not lg.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            lg.addHandler(h)
        lg.setLevel(getattr(logging, level.upper(), 20))
        return lg

# feature pipeline
try:
    from app.dataio.features.pipeline import build_feature_frame
except Exception:
    try:
        from app.dataio.features.pipeline import build_features as build_feature_frame  # fallback alias
    except Exception as _e:
        raise ImportError("Could not import build_feature_frame/build_features from app.dataio.features.pipeline") from _e

# training
try:
    from app.ml.train import train_model
except Exception:
    try:
        from app.ml.train import fit_model as train_model  # fallback alias
    except Exception as _e:
        raise ImportError("Could not import train_model/fit_model from app.ml.train") from _e

# inference wrapper
try:
    from app.ml.serve.onnx_infer import ONNXModel
except Exception:
    # Minimal shim if class name differs or file missing
    class ONNXModel:  # type: ignore
        def __init__(self, path: str):
            self.path = path
        def predict_proba(self, x: List[float]) -> float:
            # neutral probability placeholder
            return 0.5

# strategy signal factory
try:
    from app.strategy.signals import make_trade_signal
except Exception:
    try:
        from app.strategy.signals import build_signal as make_trade_signal  # fallback alias
    except Exception as _e:
        raise ImportError("Could not import make_trade_signal/build_signal from app.strategy.signals") from _e

# v0.2 backtest (new)
try:
    # You created this file already: app/orchestrator/backtest_v02.py
    from .backtest_v02 import run_backtest_v02
except Exception as _e:
    # If missing, surface a clear error only when someone tries to call it.
    def run_backtest_v02(*args, **kwargs):  # type: ignore
        raise ImportError(
            "run_backtest_v02 not available. Ensure app/orchestrator/backtest_v02.py exists "
            "and defines run_backtest_v02(cfg)."
        )


# -----------------------------------------------------------------------------
# Helpers & safety
# -----------------------------------------------------------------------------

@dataclass
class KillSwitchConfig:
    daily_loss_stop_pct: float = 5.0          # halt if day PnL <= -5%
    max_consecutive_losses: int = 3           # or 3 losses in a row


class KillSwitch:
    def __init__(self, cfg: Optional[KillSwitchConfig] = None):
        self.cfg = cfg or KillSwitchConfig()
        self._day: Optional[str] = None
        self._day_pnl_pct: float = 0.0
        self._loss_streak: int = 0

    def _ensure_day(self, day: str) -> None:
        if self._day != day:
            self._day = day
            self._day_pnl_pct = 0.0
            self._loss_streak = 0

    def on_trade(self, *, day: str, pnl_pct: float) -> bool:
        """Return True if we should HALT after applying this trade's PnL%."""
        self._ensure_day(day)
        self._day_pnl_pct += pnl_pct
        if pnl_pct < 0:
            self._loss_streak += 1
        else:
            self._loss_streak = 0
        if self._day_pnl_pct <= -abs(self.cfg.daily_loss_stop_pct):
            return True
        if self._loss_streak >= self.cfg.max_consecutive_losses:
            return True
        return False


def _infer_day(ts: Optional[int]) -> str:
    if ts is None:
        return datetime.utcnow().strftime("%Y-%m-%d")
    sec = ts / 1000.0 if ts > 10_000_000_000 else ts
    return datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class Orchestrator:
    """
    Orchestrates the end-to-end flow:
      Data (features) -> ML (train + infer) -> Strategy (signals) -> Analytics (metrics)
    """

    def __init__(self, cfg_path: str = "config/settings.example.yaml"):
        self.cfg = load_settings(cfg_path)
        # Try to read a log level if present; default INFO
        log_level = "INFO"
        try:
            lv = getattr(getattr(self.cfg, "telemetry", None), "log_level", None)
            log_level = lv or log_level
        except Exception:
            pass
        self.log = get_logger(log_level)

    # ----------------------- v0.1-compatible backtest ------------------------
    # This is left intact for backward compatibility with your earlier pipeline.

    def run_backtest(self) -> dict:
        """
        Build features, train a lightweight model, score each row to produce signals,
        and aggregate simple metrics via your v0.1 Metrics class .summary().
        """
        df = build_feature_frame(self.cfg)  # must include cfg.ml.features + cfg.ml.target
        if not isinstance(df, pd.DataFrame):
            raise TypeError("build_feature_frame() must return a pandas DataFrame")
        if df.empty:
            raise RuntimeError("Feature frame is empty. Check pipeline/horizon/data size.")

        # Resolve features/target from config in a flexible way
        try:
            features: List[str] = list(self.cfg.ml.features)
            target = self.cfg.ml.target
        except Exception:
            # attempt dict-style config
            features = list(getattr(self.cfg, "ml", {}).get("features", []))
            target = getattr(self.cfg, "ml", {}).get("target", None)

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required feature columns: {missing}")
        if target not in df.columns:
            raise RuntimeError(f"Missing target column in feature frame: {target}")

        artifact = train_model(df, self.cfg)  # should return an object with `path`
        model = ONNXModel(getattr(artifact, "path", ""))

        # Import here to avoid circulars if user upgraded metrics earlier
        from app.analytics.metrics import Metrics as V01Metrics  # v0.1-compatible

        metrics = V01Metrics()
        for _, row in df.iterrows():
            x = [float(row[f]) for f in features]
            proba_long = model.predict_proba(x)
            signal = make_trade_signal(proba_long, row)
            metrics.update(signal, row)

        summary = metrics.summary()
        self.log.info({"phase": "backtest", "summary": summary})
        return summary

    # --------------------------- v0.2 backtest -------------------------------
    # Delegates to the new functional pipeline from backtest_v02.py.

    def run_backtest_v02(
        self,
        *,
        cfg_override: Optional[object] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        v0.2 backtest entrypoint (instance method):
        - If cfg_override is provided, pass it through to the functional API.
        - Otherwise, create a lightweight proxy containing the needed fields from self.cfg.

        Any **kwargs are forwarded to run_backtest_v02 (e.g., hold_minutes, seed, output_dir override).
        """
        # Build a minimal cfg object for the functional API if not provided.
        if cfg_override is None:
            # Prefer explicit output_dir if present in kwargs; else try self.cfg.paths.output; else default.
            output_dir = kwargs.pop("output_dir", None)
            if output_dir is None:
                try:
                    output_dir = getattr(self.cfg.paths, "output", "./out/v02")
                except Exception:
                    output_dir = "./out/v02"

            # Use a SimpleNamespace-like shim without importing it to avoid extra deps here.
            class _C:
                pass
            cfg_obj = _C()
            setattr(cfg_obj, "output_dir", output_dir)

            # Optional convenience knobs you might want to propagate:
            for k in ("symbol", "start", "end", "frame", "models", "policy", "seed", "hold_minutes"):
                try:
                    v = getattr(self.cfg, k, None)
                except Exception:
                    v = None
                if v is not None:
                    setattr(cfg_obj, k, v)
        else:
            cfg_obj = cfg_override

        return run_backtest_v02(cfg_obj)

# Public exports
__all__ = ["Orchestrator", "run_backtest_v02"]
