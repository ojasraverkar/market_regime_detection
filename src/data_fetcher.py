"""
downloads OHLCV from yahoo finance
"""

import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    print(f"downloading data for {ticker} from {start} to {end}.....")
    data = yf.download(ticker, start=start, end=end, progress=False)
    prices = data['Close']

    # yfinance can return a single-column DataFrame for one ticker.
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"expected one Close column for {ticker}, got {prices.shape[1]}")
        prices = prices.iloc[:, 0]

    if prices.empty:
        raise ValueError(f"no price data returned for {ticker} between {start} and {end}")

    prices.name = 'Close'
    print(f'download for {len(prices)} complete')
    return prices

