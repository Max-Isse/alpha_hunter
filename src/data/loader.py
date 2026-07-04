import pandas as pd
from pathlib import Path
from src.utils.logging import logger


def load_data(source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load financial data from various sources.
    Handles MultiIndex columns from yfinance (2026 standard).
    """
    cache_path = Path(f"data/raw/{symbol}_{start}_{end}.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        logger.info(f"Downloading {symbol} from {source} ({start} to {end})")
        if source.lower() == "yfinance":
            import yfinance as yf
            df = yf.download(symbol, start=start, end=end, progress=False)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        df.to_parquet(cache_path)

    # --- FIX: Handle MultiIndex columns (e.g., ('Open', 'SPY')) ---
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, take the first level (Price: Open, High, etc.)
        df.columns = df.columns.get_level_values(0)
    
    # Safely standardise column names: 'open' -> 'Open'
    df.columns = [str(col)[0].upper() + str(col)[1:] for col in df.columns]
    
    return df