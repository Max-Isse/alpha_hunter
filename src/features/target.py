import pandas as pd


def compute_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward return over the specified horizon."""
    return df["Close"].pct_change(horizon).shift(-horizon)