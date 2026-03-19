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
    features = {}
    returns = log_returns(prices)
    features['return'] = returns
    
    # rolling velocity
    vol_window = config['features'].get('rolling_volatility_window')
    if vol_window:
        vol = rolling_volatility(returns, window=vol_window)
        features['volatility'] = vol
    
    df = pd.DataFrame(features).dropna()
    return df

