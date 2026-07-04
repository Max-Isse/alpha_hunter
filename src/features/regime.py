import numpy as np
import pandas as pd


def detect_regime(returns: pd.Series, window: int = 20) -> np.ndarray:
    """
    Volatility‑based regime detection.
    0 = CALM, 1 = NORMAL, 2 = VOLATILE.
    """
    vol = returns.rolling(window).std()
    vol_norm = (vol - vol.mean()) / vol.std()
    regimes = np.zeros_like(vol_norm, dtype=int)
    regimes[vol_norm < -0.5] = 0
    regimes[(vol_norm >= -0.5) & (vol_norm <= 0.5)] = 1
    regimes[vol_norm > 0.5] = 2
    return regimes