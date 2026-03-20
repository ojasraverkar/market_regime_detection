import os
import numpy as np
from src.utils import load_config
from src.data_fetcher import fetch_data
from src.features import build_feature_matrix
from src.models import train_hmm, decode_states, print_model_params
from src.visualisation import plot_regimes, scatter_features

def main():
    # load config
    config = load_config('config/config.yaml')
    print('config loaded')

    # fetch data
    prices = fetch_data(config['ticker'], config['start_date'], config['end_date'])

    # build features
    features_df = build_feature_matrix(prices, config)
    print(f'feature matrix shape: {features_df.shape}')
    print(features_df.head())

    # prepare array for hmm
    X = features_df.values

    # train model
    model_params = config['model']
    model = train_hmm(
        X,
        n_states=model_params['n_states'],
        covariance_type=model_params['covariance_type'],
        n_iter=model_params['n_iter'],
        random_state=model_params['random_state']
    )
    print('HMM training completed')

    # print model parameters
    features_names = list(features_df.columns)
    print_model_params(model, features_names)

    # decode states
    states = decode_states(model, X)

    # print state distribution
    unique, counts = np.unique(states, return_counts=True)
    for s,c in zip(unique, counts):
        print(f"state {s}: {c} days ({c/len(states)*100:.1f}%)")

    # visualize regimes on price chart
    dates = features_df.index
    prices_aligned = prices.loc[dates]

    plot_config = config['plot']
    title = f"market regime detection for {config['ticker']} ({model_params['n_states']}-state hmm with rolling volatility)"

    # create figures directory if saving
    if plot_config.get('save_figure'):
        os.makedirs('outputs/figures', exist_ok=True)
        save_path = f"outputs/figures/regimes_{config['ticker']}_{model_params['n_states']}states.png"
    else:
        save_path = None

    plot_regimes(
        prices_aligned,
        dates,
        states,
        colors=plot_config['colors'],
        title=title,
        figsize=plot_config['figsize'],
        save_path=save_path
    )   

    # optional scatter plot for features
    scatter_features(features_df, states, plot_config['colors'],
                     save_path=save_path.replace('.png', '_scatter.png') if save_path else None)
    
if __name__ == "__main__":
    main()