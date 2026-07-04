import numpy as np
import pandas as pd


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 20+ technical indicators to the DataFrame."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns
    for lag in [1, 5, 10, 21]:
        df[f"ret_{lag}"] = close.pct_change(lag)

    # Volatility
    for window in [5, 10, 21]:
        df[f"vol_{window}"] = df["ret_1"].rolling(window).std()

    # Moving averages & price ratios
    for window in [10, 21, 50]:
        df[f"ma_{window}"] = close.rolling(window).mean()
        df[f"price_ma_ratio_{window}"] = close / df[f"ma_{window}"] - 1

    # Area ratio (trend stability)
    for window in [10, 21]:
        area_above = (close - df[f"ma_{window}"]).clip(lower=0).rolling(window).sum()
        area_below = (df[f"ma_{window}"] - close).clip(lower=0).rolling(window).sum()
        df[f"area_ratio_{window}"] = area_above / (area_below + 1e-8)

    # Distribution moments
    for window in [10, 21]:
        df[f"skew_{window}"] = df["ret_1"].rolling(window).skew()
        df[f"kurt_{window}"] = df["ret_1"].rolling(window).kurt()

    # VWAP
    df["vwap"] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
    df["vwap_ratio"] = close / df["vwap"] - 1

    # OBV
    df["obv"] = (np.sign(close.diff()) * volume).cumsum()
    df["obv_ratio"] = df["obv"] / df["obv"].rolling(50).mean()

    # Price action
    df["range"] = high - low
    df["range_pct"] = df["range"] / close
    df["close_position"] = (close - low) / (high - low + 1e-8)
    df["gap"] = df["Open"] - close.shift(1)
    df["gap_pct"] = df["gap"] / close.shift(1)

    return df