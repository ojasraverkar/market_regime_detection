# Market Regime Detection for Indian Stocks

A Hidden Markov Model (HMM) based regime detection system for Indian equities, with a Streamlit dashboard.

## Features
- Downloads Indian stock data (NSE/BSE) from Yahoo Finance.
- Computes features: log returns, rolling volatility, RSI.
- Trains a Gaussian HMM with adjustable number of states (2–4).
- Walk‑forward validation with transaction costs and turnover analysis.
- Interactive Streamlit dashboard to explore regimes in real time.

## Reports
- [Full project report (Markdown)](REPORT.md)
- [PDF version](report.pdf) 

## Quick Start
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`

## Configuration
Adjust `config/config.yaml` for different stocks, date ranges, and model parameters.

## Results Summary
- The HMM identifies three interpretable regimes (low‑volatility calm, high‑volatility bullish, high‑volatility bearish).
- Gross Sharpe improvements possible with careful training‑window selection.
- High turnover (daily rebalancing) leads to significant transaction cost drag.
- Best suited for regime monitoring and risk filtering rather than standalone trading.

