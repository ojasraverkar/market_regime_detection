"""
computes features from price series
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series, scale: float = 100.0) -> pd.Series:
    # scale is multiplicative factor, 100 for percentage
    returns = scale * np.log(prices / prices.shift(1))
    return returns.dropna()


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.fillna(100)
    return 100 - (100 / (1 + rs))


def build_feature_matrix(prices: pd.Series, config: dict) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(f"expected a single price series, got shape {prices.shape}")
        prices = prices.iloc[:, 0]

    feature_config = config['features']
    feature_frames = {}

    returns = log_returns(prices).rename('return')
    feature_frames['return'] = returns

    vol_window = feature_config.get('rolling_volatility_window')
    if vol_window:
        feature_frames['volatility'] = rolling_volatility(returns, window=vol_window).rename('volatility')

    rsi_window = feature_config.get('rsi_window')
    if rsi_window:
        rsi_vals = rsi(prices, window=rsi_window).rename('rsi')
        feature_frames['rsi'] = rsi_vals.loc[returns.index]

    df = pd.concat(feature_frames.values(), axis=1).dropna()
    return df
