import numpy as np
from src.backtest.metrics import calculate_metrics


def test_metrics():
    returns = np.random.randn(252) * 0.01
    metrics = calculate_metrics(returns)
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics