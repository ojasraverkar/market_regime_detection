"""
creates plots
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

def plot_regimes(prices:pd.Series, dates: pd.DatetimeIndex, states: np.ndarray, colors: list, title:str, 
                 figsize: tuple = (14, 7), save_path: str = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, prices, color = 'black', linewidth = 1, label = 'Price')

    for i, state in enumerate(states):
        start = dates[i]
        end = dates[i+1] if i+1 < len(dates) else dates[i]
        ax.axvspan(start, end, alpha = 0.3, color=colors[state])

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close price INR')
    ax.legend(loc = 'upper left')

    # covid crash
    ax.axvline(pd.Timestamp('2020-03-01'), color='grey', linestyle='--', alpha = 0.7, label='covid crash start')
    ax.legend(loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"plot saved to {save_path}")
    plt.show()

def scatter_features(features: pd.DataFrame, states: np.ndarray, colors: list, 
                     title: str ="Feature space coloured by regime", save_path: str = None):
    if features.shape[1] != 2:
        print('scatter plot requires exactly 2 features, requirement not met')
        return
    
    fig, ax = plt.subplots(figsize = (10,6))
    scatter = ax.scatter(features.iloc[:, 0], features.iloc[:, 1], c=states, cmap='bwr', alpha = 0.6, 
                         edgecolors='k', linewidth = 0.5)
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    ax.set_title(title)
    plt.colorbar(scatter, label='State')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 150)
    plt.show()
