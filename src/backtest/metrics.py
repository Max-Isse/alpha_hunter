import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Calculate key performance metrics."""
    excess = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess.mean() / returns.std() if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    sortino = np.sqrt(252) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0.0

    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    drawdown = (peak - cum) / peak
    max_dd = drawdown.max()

    return {
        "total_return": cum.iloc[-1] - 1,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility": returns.std() * np.sqrt(252),
    }