import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.backtest.metrics import calculate_metrics
from src.utils.logging import logger


def run_backtest(predictions: np.ndarray, actuals: np.ndarray, config) -> pd.DataFrame:
    """
    Simple vectorised backtest.
    Assumes predictions and actuals are aligned daily.
    """
    position = np.where(predictions > 0, config.position_size, 0.0)
    # Apply slippage (price impact)
    returns = actuals * (1 - config.slippage * np.abs(position)) - config.commission * np.abs(np.diff(np.concatenate([[0], position])))
    portfolio = config.initial_capital * (1 + returns).cumprod()

    metrics = calculate_metrics(pd.Series(returns))
    logger.info(f"Backtest results: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:.2%}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio, label="Strategy")
    bench = config.initial_capital * (1 + actuals).cumprod()
    plt.plot(bench, label="Buy & Hold", alpha=0.7)
    plt.legend()
    plt.title("Backtest Equity Curve")
    plt.grid(alpha=0.3)
    plt.savefig("results/plots/equity_curve.png", dpi=150)
    plt.close()

    return pd.DataFrame({"portfolio": portfolio, "returns": returns})