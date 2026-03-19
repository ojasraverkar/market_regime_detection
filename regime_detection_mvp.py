"""
Market Regime Detection model (first iteration) 
- uses daily adjusted  close prices from yfinance
- computed log returns
- trains 2-state Gaussian HMM
- plots price with regime coloured background
"""

# imports
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

# downloading
ticker = "ETERNAL.NS"
start_date = "2021-01-01"
end_date = "2026-03-19"

print(f"downloading data for {ticker} from {start_date} ~ ~ ~")
data = yf.download(ticker, start=start_date, end=end_date)
prices = data['Close']
print(f"downloaded {len(prices)} days of data")

# verify download
'''
print("First 5 rows:")
print(prices.head())
'''

# log return = 100 * ln(P_t / P_{t-1}
# multiplied by 100 just for comfort

returns = 100 * np.log(prices / prices.shift(1)).dropna()

print(f"\ncomputed {len(returns)} daily log returns.")
# verify
'''
print(returns.head())
'''

# hmmlearn expects 2D array (n_samples, n_features)
# we have 1 feature (returns) so reshaped as (n_days, 1)
X = returns.values.reshape(-1,1)
print(f"\nfeature array shape: {X.shape}")

# gaussian hmm with 2 states
# covariance_type="full" means each state has its own full covariance matrix
# since we have only 1 feature, it's just variance
# n_iter is the maximum number of em iterations.
# random_state ensures reproducibility.
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)

print("\ntraining HMM ~ ~ ~")
model.fit(X)
print("training completed")

print("\n=== learned model aarameters ===")
print("transition matrix (rows=from state, cols=to state):")
print(model.transmat_)
print()

for i in range(model.n_components):
    mean = model.means_[i][0]
    std = np.sqrt(model.covars_[i][0][0])
    print(f"State {i}: mean return = {mean:.2f}%, volatility = {std:.2f}%")

# viterbi algorithm finds the single best state path.
hidden_states = model.predict(X)

print(f"\nDecoded states for each day (first 10): {hidden_states[:10]}")

# count how many days in each state
unique, counts = np.unique(hidden_states, return_counts=True)
for state, count in zip(unique, counts):
    print(f"State {state}: {count} days ({count/len(hidden_states)*100:.1f}%)")

# we need to align the dates with the states.
# Note: returns index (dates) starts from the second day because we lost the first day when shifting.
dates = returns.index

# create the plot
fig, ax = plt.subplots(figsize=(14, 7))

# plot the price line (using dates that correspond to returns)
ax.plot(dates, prices.loc[dates], color='black', linewidth=1, label=f'{ticker} Price')

# color the background according to the state
colors = ['blue', 'red']   # blue for state 0, red for state 1 (you can swap based on interpretation)
for i, state in enumerate(hidden_states):
    # draw a vertical span from this date to the next date (or the same if last)
    start = dates[i]
    end = dates[i+1] if i+1 < len(dates) else dates[i]
    ax.axvspan(start, end, alpha=0.3, color=colors[state])

# add labels and title
ax.set_title(f'Market Regime Detection for {ticker} (2-State HMM)')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted Close Price (INR)')
ax.legend(loc='upper left')

# Optional: add vertical lines at major events for reference
# e.g., COVID crash in March 2020
# ax.axvline(pd.Timestamp('2020-03-01'), color='gray', linestyle='--', alpha=0.7, label='COVID crash start')

plt.tight_layout()
plt.show()

print("\nInterpretation hint:")
print("- Look at which state has higher volatility (std). That state likely captures turbulent periods.")
print("- Check if the high-volatility state (red if we set it so) appears during known crises (e.g., March 2020).")
print("- If states flip too often, the model might be overfitting noise. We can later add smoothing.")