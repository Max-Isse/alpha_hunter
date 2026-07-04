import pandas as pd
from pathlib import Path
from src.utils.logging import logger


def load_data(source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load financial data from various sources.
    Supports 'yfinance' and local CSV cache.
    """
    cache_path = Path(f"data/raw/{symbol}_{start}_{end}.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading {symbol} from {source} ({start} to {end})")
    if source.lower() == "yfinance":
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    # Standardise column names
    df.columns = [col[0].upper() + col[1:] for col in df.columns]  # 'open' -> 'Open'
    df.to_parquet(cache_path)
    return df