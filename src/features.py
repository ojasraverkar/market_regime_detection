"""
computes features from price series
"""

import numpy as np
import pandas as pd

def log_returns(prices: pd.Series, scale: float = 100.0) -> pd.Series: #scale is multiplicative factor, 100 for percentage
    returns = scale * np.log(prices / prices.shift(1))
    return returns.dropna()

def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()

def build_feature_matrix(prices: pd.Series, config: dict) ->pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"expected a single price series, got shape {prices.shape}")
        prices = prices.iloc[:, 0]

    features = []
    returns = log_returns(prices).rename('return')
    features.append(returns)
    
    # rolling volatility
    vol_window = config['features'].get('rolling_volatility_window')
    if vol_window:
        vol = rolling_volatility(returns, window=vol_window).rename('volatility')
        features.append(vol)
    
    df = pd.concat(features, axis=1).dropna()
    return df

