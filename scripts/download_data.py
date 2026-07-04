import click
from src.data.loader import load_data


@click.command()
@click.option("--symbol", default="SPY")
@click.option("--start", default="2010-01-01")
@click.option("--end", default="2026-06-01")
def download(symbol, start, end):
    load_data("yfinance", symbol, start, end)
    print(f"Downloaded {symbol} data.")


if __name__ == "__main__":
    download()