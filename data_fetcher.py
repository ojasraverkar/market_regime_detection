import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# downloading
ticker = "RELIANCE.NS"
data = yf.download(tickers=ticker, start='2015-01-01', end='2025-01-01')
prices = data['Close']

# verfifying
print(prices.head())
print(prices.tail())