import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from src.features.technical import compute_technical_features
from src.features.target import compute_target
from src.utils.logging import logger


class FeaturePipeline:
    """
    Leakage‑proof feature engineering pipeline.
    Uses src.features for modular feature generation.
    """
    def __init__(self, lookback: int = 60, horizon: int = 5):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_fe = compute_technical_features(df)
        target = compute_target(df_fe, self.horizon)

        # Drop NaN and align
        df_clean = df_fe.dropna()
        target_clean = target.loc[df_clean.index]

        exclude_cols = ["Open", "High", "Low", "Close", "Volume", "vwap", "obv"]
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols and c not in ["Date"]]
        self.feature_columns = feature_cols

        X_raw = df_clean[feature_cols].values
        X_scaled = self.scaler.fit_transform(X_raw)

        X, y = [], []
        for i in range(self.lookback, len(X_scaled)):
            X.append(X_scaled[i - self.lookback:i])
            y.append(target_clean.iloc[i])

        logger.info(f"Created {len(X)} sequences with {len(feature_cols)} features each.")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_fe = compute_technical_features(df)
        df_clean = df_fe.dropna()
        X_raw = df_clean[self.feature_columns].values
        X_scaled = self.scaler.transform(X_raw)
        return X_scaled[-self.lookback:].reshape(1, self.lookback, -1)