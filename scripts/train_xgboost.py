"""
Train an XGBoost classifier for directional prediction.
Uses the same features as the Transformer model but treats the problem as classification.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pathlib import Path

from src.utils.config import AppConfig
from src.utils.logging import setup_logging, logger
from src.data.loader import load_data
from src.features.technical import compute_technical_features
from src.features.target import compute_target
from src.backtest import run_backtest


def main():
    setup_logging(level="INFO")
    config = AppConfig()
    
    # 1. Load data
    logger.info(f"Loading {config.data.symbol}...")
    df = load_data(config.data.source, config.data.symbol, config.data.start, config.data.end)
    
    # 2. Feature engineering (flat features, no sequences)
    logger.info("Computing features...")
    df_fe = compute_technical_features(df)
    
    # 3. Target: classification (1 if forward return > 0 else 0)
    target = compute_target(df_fe, config.model.horizon, as_classification=True)
    
    # 4. Align and clean
    df_clean = df_fe.dropna()
    target_clean = target.loc[df_clean.index]
    
    # 5. Feature columns (exclude price/volume columns that are not features)
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "vwap", "obv"]
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols and c not in ["Date"]]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df_clean[feature_cols].values
    y = target_clean.values
    
    # 6. Chronological split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 7. Train XGBoost
    logger.info("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 8. Evaluate on test set
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_class)
    prec = precision_score(y_test, y_pred_class)
    rec = recall_score(y_test, y_pred_class)
    logger.info(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # 9. Backtest
    # Use prediction probability as "prediction" (higher = more confident up)
    # We need to align with the test set dates (we'll pass the probability)
    logger.info("Running backtest...")
    results = run_backtest(
        y_pred_prob,      # predictions (probability of up)
        y_test,           # actuals (binary: 1=up, 0=down)
        config.backtest,
        horizon=config.model.horizon
    )
    results.to_csv("results/xgboost_backtest.csv", index=False)
    logger.info("XGBoost pipeline complete.")


if __name__ == "__main__":
    main()