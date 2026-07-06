import pandas as pd

def compute_target(df: pd.DataFrame, horizon: int, as_classification: bool = False) -> pd.Series:
    """
    Compute forward return over horizon.
    If as_classification=True, return 1 if return > 0 else 0.
    """
    forward_ret = df["Close"].pct_change(horizon).shift(-horizon)
    if as_classification:
        return (forward_ret > 0).astype(int)
    return forward_ret