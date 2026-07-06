"""
Train an XGBoost Regressor for return prediction.
Trades only when the predicted return is in the top/bottom 30%.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    
    # 2. Feature engineering
    logger.info("Computing features...")
    df_fe = compute_technical_features(df)
    
    # 3. Target: raw forward return (regression)
    target = compute_target(df_fe, config.model.horizon, as_classification=False)
    
    # 4. Align features and target by dropping NaN from both
    # Combine into a single DataFrame for easy filtering
    combined = df_fe.copy()
    combined['target'] = target
    
    # Drop rows where any feature or target is NaN
    combined = combined.dropna()
    
    # Separate features and target
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "vwap", "obv", "target"]
    feature_cols = [c for c in combined.columns if c not in exclude_cols and c not in ["Date"]]
    logger.info(f"Using {len(feature_cols)} features")
    
    X = combined[feature_cols].values
    y = combined['target'].values
    
    # 5. Chronological split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Ensure no NaN in test set (should be guaranteed by dropna)
    if np.isnan(y_test).any():
        raise ValueError("y_test contains NaN values!")
    if np.isnan(X_test).any():
        raise ValueError("X_test contains NaN values!")
    
    # 6. Train XGBoost Regressor
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 7. Evaluate on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # 8. Correlation
    corr = np.corrcoef(y_pred, y_test)[0, 1]
    logger.info(f"Test correlation: {corr:.4f}")
    
    # 9. Backtest
    logger.info("Running backtest...")
    results = run_backtest(
        y_pred,           # predictions (continuous returns)
        y_test,           # actuals
        config.backtest,
        horizon=config.model.horizon
    )
    results.to_csv("results/xgboost_regressor_backtest.csv", index=False)
    logger.info("XGBoost Regressor pipeline complete.")


if __name__ == "__main__":
    main()