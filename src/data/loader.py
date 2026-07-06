import pandas as pd
import time
from pathlib import Path
from src.utils.logging import logger


def load_data_with_vix(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df_stock = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    df_vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    # Merge on date index
    df_merged = df_stock.join(df_vix['Close'].rename('VIX'), how='inner')
    return df_merged

def load_data(source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load financial data with robust column standardisation.
    """
    
    cache_path = Path(f"data/raw/{symbol}_{start}_{end}.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try cache
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        if not df.empty:
            # Ensure columns are standardised even if cached
            df = standardise_columns(df)
            return df
        else:
            logger.warning("Cached file is empty, re-downloading...")
            cache_path.unlink()

    # 2. Download with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {symbol} from {source} ({start} to {end}) - attempt {attempt+1}")
            if source.lower() == "yfinance":
                import yfinance as yf
                # Use auto_adjust=True to avoid MultiIndex
                df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            else:
                raise ValueError(f"Unsupported data source: {source}")

            if df.empty:
                raise ValueError("Empty DataFrame returned by yfinance")

            # Standardise columns to ['Open', 'High', 'Low', 'Close', 'Volume']
            df = standardise_columns(df)

            # Save to cache
            df.to_parquet(cache_path)
            logger.info(f"Download successful. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download data after {max_retries} attempts.")

    raise RuntimeError("Unable to load data.")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has columns: Open, High, Low, Close, Volume.
    Handles MultiIndex and case variations.
    """
    # If MultiIndex, flatten to first level (in case of single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Map any common names to our standard
    column_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'close', 'high', 'low', 'volume']:
            column_map[col] = col_lower.capitalize()  # Open, Close, etc.
    if column_map:
        df = df.rename(columns=column_map)
    else:
        # If no mapping found, raise error
        raise ValueError("No recognised column names found. Expected: Open, High, Low, Close, Volume")

    # Ensure we have all required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df[required]  # keep only needed columns