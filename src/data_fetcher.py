"""
downloads OHLCV from yahoo finance
"""

import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start: str, end: str) -> pd.Series:
    print(f"downloading data for {ticker} from {start} to {end}.....")
    data = yf.download(ticker, start=start, end=end, progress=False)
    prices = data['Close']
    print(f'download for {len(prices)} complete')
    return prices

