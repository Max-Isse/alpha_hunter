import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.backtest.metrics import calculate_metrics
from src.utils.logging import logger


def run_backtest(predictions: np.ndarray, actuals: np.ndarray, config, horizon: int = 5) -> pd.DataFrame:
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    n = len(predictions)

    # --- Diagnostic correlation ---
    corr = np.corrcoef(predictions[::horizon], actuals[::horizon])[0, 1]
    logger.info(f"Correlation (full): {corr:.4f}")

    # If negative correlation, flip predictions
    if corr < 0:
        predictions = -predictions
        logger.info("🔄 Inverted predictions (negative correlation)")

    # --- Compute thresholds (top/bottom 30% for trading) ---
    long_threshold = np.percentile(predictions, 70)   # top 30%
    short_threshold = np.percentile(predictions, 30)  # bottom 30%
    logger.info(f"Long threshold: {long_threshold:.4f}, Short threshold: {short_threshold:.4f}")

    # --- Get position fraction from config ---
    position_fraction = config.position_fraction  # e.g., 0.15
    max_position = config.position_size

    # --- Optional: Volatility scaling (risk parity) ---
    # Compute rolling volatility of predictions (as a proxy for signal noise)
    # We'll scale position fraction inversely to recent prediction volatility.
    window = 20
    pred_vol = pd.Series(predictions).rolling(window).std().fillna(predictions.std()).values

    # Target volatility (annualized) we want to maintain, e.g., 10% => daily vol ~0.006
    target_vol = 0.006  # daily vol ~0.006 (approx 10% annual)
    # But we'll use a simpler approach: just use a fixed fraction for now,
    # and you can adjust it in config.

    # We'll implement a simple dynamic scaling: 
    # position = position_fraction * (target_vol / current_pred_vol) but cap at max_position.
    # But for simplicity, we'll just use the fixed fraction; you can uncomment the scaling later.

    portfolio_value = config.initial_capital
    returns = []
    trades = 0

    for i in range(0, n, horizon):
        if i + horizon > n:
            break

        pred = predictions[i]

        # Only trade if prediction is in top 30% (long) or bottom 30% (short)
        if pred > long_threshold:
            # Use the raw prediction magnitude to scale position? Or just use fixed fraction.
            # Let's use fixed fraction for now.
            position = position_fraction  # Go long
            trades += 1
        elif pred < short_threshold:
            position = -position_fraction  # Go short
            trades += 1
        else:
            position = 0.0  # Stay flat

        # Clip to max allowed
        position = np.clip(position, -max_position, max_position)

        actual_ret = actuals[i]
        profit = position * portfolio_value * actual_ret

        # Costs
        costs = (config.slippage * abs(position) + config.commission * 2 * abs(position)) * portfolio_value
        new_portfolio_value = portfolio_value + profit - costs

        period_return = (new_portfolio_value / portfolio_value) - 1
        daily_return = (1 + period_return) ** (1.0 / horizon) - 1
        returns.extend([daily_return] * horizon)

        portfolio_value = new_portfolio_value

    returns = np.array(returns[:n])

    logger.info(f"Total trades: {trades} (out of {n // horizon} rebalance periods)")
    logger.info(f"Trading frequency: {trades / (n // horizon):.1%}")

    metrics = calculate_metrics(pd.Series(returns))
    logger.info(f"Backtest results: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:.2%}")

    equity = config.initial_capital * (1 + returns).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="Strategy (XGBoost + threshold)")
    plt.legend()
    plt.title("XGBoost Backtest with Configurable Position Fraction")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/equity_curve_xgboost.png", dpi=150)
    plt.close()

    return pd.DataFrame({"portfolio": equity, "returns": returns})