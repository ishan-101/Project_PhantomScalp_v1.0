# app/dataio/features/pipeline.py
import numpy as np
import pandas as pd
from app.common.config import Settings
from .indicators import compute_indicators

def build_feature_frame(cfg: Settings, n_rows: int = 300, horizon: int = 2) -> pd.DataFrame:
    """
    Build a synthetic feature frame with enough rows so that the shifted target
    doesn't drop everything. Returns a DataFrame containing *at least* the
    features in cfg.ml.features and the target column.
    """
    rng = np.random.default_rng(42)

    # Synthetic OHLCV-ish series
    close = 100 + np.cumsum(rng.normal(0, 0.25, n_rows))
    volume = rng.integers(10, 30, n_rows)

    df = pd.DataFrame({"close": close, "volume": volume})

    # Add indicators: must create rsi_14, ema_20, obv, micro_price_imbalance
    df = compute_indicators(df)

    # Forward target over a short horizon so we keep plenty of rows
    tgt = cfg.ml.target
    df[tgt] = df["close"].pct_change(periods=horizon).shift(-horizon)

    # Keep rows where target and features are all present
    needed_cols = list(cfg.ml.features) + [tgt]
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    # Sanity: ensure all required features exist
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features from feature frame: {missing}")

    return df[needed_cols]
