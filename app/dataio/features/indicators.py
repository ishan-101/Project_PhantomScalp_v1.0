# app/dataio/features/indicators.py
import pandas as pd

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMA 20
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()

    # RSI 14 (simple implementation)
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["rsi_14"] = out["rsi_14"].fillna(50)

    # OBV
    sign = (out["close"].diff().fillna(0).gt(0).astype(int) * 2 - 1)
    out["obv"] = (sign * out["volume"]).cumsum().astype(float)

    # Micro price imbalance: close - SMA5
    sma5 = out["close"].rolling(5, min_periods=1).mean()
    out["micro_price_imbalance"] = out["close"] - sma5

    return out
