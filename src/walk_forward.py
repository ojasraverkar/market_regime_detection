"""
walk-forward validation and strategy comparison for regime probabilities
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .utils import load_config
    from .data_fetcher import fetch_data
    from .features import build_feature_matrix
    from .models import train_hmm_on_slice, decode_states
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils import load_config
    from src.data_fetcher import fetch_data
    from src.features import build_feature_matrix
    from src.models import train_hmm_on_slice, decode_states

TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRAIN_WINDOWS = [750, 1000, 1250, 1500]
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7]
DEFAULT_TRANSACTION_COST_BPS = 10.0


def walk_forward_validation(features_df, model_config, train_window_days=1250, test_window_days=20):
    n = len(features_df)
    predicted_states = []
    predicted_probs = []
    actual_dates = []

    if n <= train_window_days + test_window_days:
        raise ValueError(
            f"not enough feature rows ({n}) for train window {train_window_days} and test window {test_window_days}"
        )

    for start in range(0, n - train_window_days - test_window_days + 1, test_window_days):
        train_end = start + train_window_days
        test_start = train_end
        test_end = test_start + test_window_days

        train_features = features_df.iloc[start:train_end].values
        test_features = features_df.iloc[test_start:test_end].values
        test_dates = features_df.index[test_start:test_end]

        model = train_hmm_on_slice(
            train_features,
            n_states=model_config['n_states'],
            covariance_type=model_config['covariance_type'],
            n_iter=model_config['n_iter'],
            random_state=model_config['random_state'],
        )

        bullish_state = int(np.argmax(model.means_[:, 0]))
        states = decode_states(model, test_features)
        bullish_probs = model.predict_proba(test_features)[:, bullish_state]

        predicted_states.extend(states.tolist())
        predicted_probs.extend(bullish_probs.tolist())
        actual_dates.extend(test_dates)

        print(
            f"trained on {train_features.shape[0]} days, predicted {test_features.shape[0]} days"
            + f" bullish state: {bullish_state}"
        )

    return np.array(predicted_states), np.array(predicted_probs), pd.DatetimeIndex(actual_dates)


def build_probability_exposure(probs):
    probs = np.asarray(probs)
    exposure = np.zeros(len(probs))
    if len(probs) > 1:
        exposure[1:] = probs[:-1]
    return exposure


def build_threshold_exposure(probs, threshold):
    probs = np.asarray(probs)
    exposure = np.zeros(len(probs))
    if len(probs) > 1:
        exposure[1:] = (probs[:-1] >= threshold).astype(float)
    return exposure


def backtest_from_exposure(prices, dates, exposure, transaction_cost_bps=0.0):
    prices_aligned = prices.loc[dates]
    asset_returns = prices_aligned.pct_change().fillna(0).values
    exposure = np.asarray(exposure)
    turnover = np.abs(np.diff(exposure, prepend=0.0))
    transaction_cost_rate = transaction_cost_bps / 10000.0
    transaction_costs = turnover * transaction_cost_rate
    gross_returns = exposure * asset_returns
    net_returns = gross_returns - transaction_costs

    return {
        'asset_returns': asset_returns,
        'exposure': exposure,
        'turnover': turnover,
        'transaction_costs': transaction_costs,
        'gross_returns': gross_returns,
        'net_returns': net_returns,
    }


def buy_and_hold_returns(prices, dates):
    return prices.loc[dates].pct_change().fillna(0).values


def cumulative_returns(returns):
    return (1 + np.asarray(returns)).cumprod()


def compute_sharpe_ratio(returns, annualization_factor=TRADING_DAYS_PER_YEAR):
    returns = np.asarray(returns)
    if len(returns) < 2:
        return np.nan

    volatility = returns.std(ddof=1)
    if np.isclose(volatility, 0.0):
        return np.nan

    return np.sqrt(annualization_factor) * returns.mean() / volatility


def summarize_strategy(name, backtest_result, benchmark_returns, threshold=np.nan):
    gross_curve = cumulative_returns(backtest_result['gross_returns'])
    net_curve = cumulative_returns(backtest_result['net_returns'])
    benchmark_curve = cumulative_returns(benchmark_returns)
    gross_sharpe = compute_sharpe_ratio(backtest_result['gross_returns'])
    net_sharpe = compute_sharpe_ratio(backtest_result['net_returns'])
    benchmark_sharpe = compute_sharpe_ratio(benchmark_returns)

    return {
        'strategy_name': name,
        'threshold': threshold,
        'strategy_gross_cumulative_return': float(gross_curve[-1] - 1),
        'strategy_net_cumulative_return': float(net_curve[-1] - 1),
        'buy_hold_cumulative_return': float(benchmark_curve[-1] - 1),
        'strategy_gross_sharpe': float(gross_sharpe) if not np.isnan(gross_sharpe) else np.nan,
        'strategy_net_sharpe': float(net_sharpe) if not np.isnan(net_sharpe) else np.nan,
        'buy_hold_sharpe': float(benchmark_sharpe) if not np.isnan(benchmark_sharpe) else np.nan,
        'turnover_total': float(backtest_result['turnover'].sum()),
        'avg_daily_turnover': float(backtest_result['turnover'].mean()),
        'total_transaction_cost': float(backtest_result['transaction_costs'].sum()),
        'cost_drag': float(gross_curve[-1] - net_curve[-1]),
        'gross_curve': gross_curve,
        'net_curve': net_curve,
        'buy_hold_curve': benchmark_curve,
        'net_returns': backtest_result['net_returns'],
        'gross_returns': backtest_result['gross_returns'],
        'exposure': backtest_result['exposure'],
        'turnover': backtest_result['turnover'],
        'transaction_costs': backtest_result['transaction_costs'],
        'buy_hold_returns': np.asarray(benchmark_returns),
    }


def ensure_output_dirs():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)


def plot_out_of_sample_regimes(prices, states, dates, colors, title, save_path=None, figsize=(14, 7)):
    _, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, prices.loc[dates], color='black', linewidth=1, label='Price')

    max_state = int(np.max(states)) if len(states) else -1
    if len(colors) <= max_state:
        raise ValueError(f"not enough colors for states: need {max_state + 1}, got {len(colors)}")

    for i, state in enumerate(states):
        start = dates[i]
        end = dates[i + 1] if i + 1 < len(dates) else dates[i]
        ax.axvspan(start, end, alpha=0.3, color=colors[state])

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"plot saved to {save_path}")
    plt.show()


def plot_strategy_comparison(dates, strategy_curves, bh_curve, title, save_path=None, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    for label, curve in strategy_curves.items():
        plt.plot(dates, curve, label=label)
    plt.plot(dates, bh_curve, label='Buy & Hold', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"plot saved to {save_path}")
    plt.show()


def plot_bullish_probability(dates, probs, title, save_path=None, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    plt.plot(dates, probs, color='darkorange', linewidth=1.2)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Bullish Probability')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"plot saved to {save_path}")
    plt.show()


def save_walk_forward_report(dates, states, probs, strategy_results, output_path):
    report_df = pd.DataFrame(
        {
            'date': dates,
            'predicted_state': states,
            'bullish_probability': probs,
            'buy_hold_return': strategy_results['buy_hold']['buy_hold_returns'],
            'probability_weighted_exposure': strategy_results['probability_weighted']['exposure'],
            'probability_weighted_gross_return': strategy_results['probability_weighted']['gross_returns'],
            'probability_weighted_net_return': strategy_results['probability_weighted']['net_returns'],
            'probability_weighted_turnover': strategy_results['probability_weighted']['turnover'],
            'probability_weighted_transaction_cost': strategy_results['probability_weighted']['transaction_costs'],
        }
    )

    for name, result in strategy_results.items():
        if name in {'buy_hold', 'probability_weighted'}:
            continue
        report_df[f"{name}_exposure"] = result['exposure']
        report_df[f"{name}_gross_return"] = result['gross_returns']
        report_df[f"{name}_net_return"] = result['net_returns']
        report_df[f"{name}_turnover"] = result['turnover']
        report_df[f"{name}_transaction_cost"] = result['transaction_costs']

    report_df.to_csv(output_path, index=False)
    print(f"report saved to {output_path}")


def save_sweep_results(results_df, output_path):
    results_df.to_csv(output_path, index=False)
    print(f"sweep results saved to {output_path}")


def evaluate_strategies(prices, probs, dates, thresholds, transaction_cost_bps):
    bh_returns = buy_and_hold_returns(prices, dates)
    results = {
        'probability_weighted': summarize_strategy(
            'probability_weighted',
            backtest_from_exposure(
                prices,
                dates,
                build_probability_exposure(probs),
                transaction_cost_bps=transaction_cost_bps,
            ),
            bh_returns,
        ),
        'buy_hold': {
            'strategy_name': 'buy_hold',
            'buy_hold_returns': bh_returns,
            'buy_hold_curve': cumulative_returns(bh_returns),
            'buy_hold_sharpe': compute_sharpe_ratio(bh_returns),
        },
    }

    for threshold in thresholds:
        name = f"threshold_{threshold:.2f}"
        results[name] = summarize_strategy(
            name,
            backtest_from_exposure(
                prices,
                dates,
                build_threshold_exposure(probs, threshold),
                transaction_cost_bps=transaction_cost_bps,
            ),
            bh_returns,
            threshold=threshold,
        )

    return results


def strategy_results_table(
    strategy_results,
    train_window_days,
    test_window_days,
    avg_bullish_probability,
    transaction_cost_bps,
):
    rows = []
    for name, result in strategy_results.items():
        if name == 'buy_hold':
            continue
        rows.append(
            {
                'strategy_name': result['strategy_name'],
                'train_window_days': train_window_days,
                'test_window_days': test_window_days,
                'transaction_cost_bps': transaction_cost_bps,
                'strategy_gross_cumulative_return': result['strategy_gross_cumulative_return'],
                'strategy_net_cumulative_return': result['strategy_net_cumulative_return'],
                'buy_hold_cumulative_return': result['buy_hold_cumulative_return'],
                'strategy_gross_sharpe': result['strategy_gross_sharpe'],
                'strategy_net_sharpe': result['strategy_net_sharpe'],
                'buy_hold_sharpe': result['buy_hold_sharpe'],
                'avg_bullish_probability': avg_bullish_probability,
                'threshold': result['threshold'],
                'turnover_total': result['turnover_total'],
                'avg_daily_turnover': result['avg_daily_turnover'],
                'total_transaction_cost': result['total_transaction_cost'],
                'cost_drag': result['cost_drag'],
            }
        )
    return pd.DataFrame(rows).sort_values('strategy_net_sharpe', ascending=False).reset_index(drop=True)


def print_strategy_results(results_df):
    for row in results_df.itertuples(index=False):
        print(f"{row.strategy_name} gross cumulative return: {row.strategy_gross_cumulative_return:.2%}")
        print(f"{row.strategy_name} net cumulative return: {row.strategy_net_cumulative_return:.2%}")
        print(f"buy and hold cumulative return: {row.buy_hold_cumulative_return:.2%}")
        print(f"{row.strategy_name} gross sharpe ratio: {row.strategy_gross_sharpe:.3f}")
        print(f"{row.strategy_name} net sharpe ratio: {row.strategy_net_sharpe:.3f}")
        print(f"buy and hold sharpe ratio: {row.buy_hold_sharpe:.3f}")
        print(f"{row.strategy_name} total turnover: {row.turnover_total:.2f}")
        print(f"{row.strategy_name} total transaction cost: {row.total_transaction_cost:.2%}")
        print()


def run_single_walk_forward(
    config,
    train_window_days=None,
    test_window_days=None,
    thresholds=None,
    transaction_cost_bps=None,
    show_plots=True,
):
    prices = fetch_data(config['ticker'], config['start_date'], config['end_date'])
    features_df = build_feature_matrix(prices, config)
    model_config = config['model']
    walk_forward_config = config.get('walk_forward', {})
    plot_config = config.get('plot', {})

    train_window_days = train_window_days or walk_forward_config.get('train_window_days', 1250)
    test_window_days = test_window_days or walk_forward_config.get('test_window_days', 20)
    thresholds = thresholds or walk_forward_config.get('thresholds', DEFAULT_THRESHOLDS)
    transaction_cost_bps = (
        transaction_cost_bps
        if transaction_cost_bps is not None
        else walk_forward_config.get('transaction_cost_bps', DEFAULT_TRANSACTION_COST_BPS)
    )
    price_figsize = tuple(plot_config.get('figsize', [14, 7]))

    states, probs, dates = walk_forward_validation(
        features_df,
        model_config,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )

    strategy_results = evaluate_strategies(prices, probs, dates, thresholds, transaction_cost_bps)
    results_df = strategy_results_table(
        strategy_results,
        train_window_days,
        test_window_days,
        avg_bullish_probability=float(np.mean(probs)),
        transaction_cost_bps=transaction_cost_bps,
    )

    print_strategy_results(results_df)
    print('strategy comparison:')
    print(results_df.to_string(index=False))

    ticker_slug = config['ticker'].replace('.', '_')
    if plot_config.get('save_figure'):
        ensure_output_dirs()
        save_walk_forward_report(
            dates,
            states,
            probs,
            strategy_results,
            f"outputs/reports/walk_forward_{ticker_slug}_{train_window_days}_{test_window_days}.csv",
        )
        comparison_path = (
            f"outputs/reports/walk_forward_comparison_{ticker_slug}_{train_window_days}_{test_window_days}.csv"
        )
        results_df.to_csv(comparison_path, index=False)
        print(f"comparison saved to {comparison_path}")

        if show_plots:
            plot_out_of_sample_regimes(
                prices,
                states,
                dates,
                plot_config['colors'],
                title=f"Out-of-Sample Regimes for {config['ticker']}",
                save_path=f"outputs/figures/walk_forward_regimes_{ticker_slug}_{train_window_days}_{test_window_days}.png",
                figsize=price_figsize,
            )
            strategy_curves = {
                row.strategy_name: strategy_results[row.strategy_name]['net_curve']
                for row in results_df.itertuples(index=False)
            }
            plot_strategy_comparison(
                dates,
                strategy_curves,
                strategy_results['buy_hold']['buy_hold_curve'],
                title=f"Walk-Forward Strategy Comparison for {config['ticker']} (Net of Costs)",
                save_path=f"outputs/figures/walk_forward_equity_{ticker_slug}_{train_window_days}_{test_window_days}.png",
            )
            plot_bullish_probability(
                dates,
                probs,
                title=f"Bullish State Probability for {config['ticker']}",
                save_path=f"outputs/figures/walk_forward_probs_{ticker_slug}_{train_window_days}_{test_window_days}.png",
            )

    return results_df


def run_train_window_sweep(
    config,
    train_windows=None,
    test_window_days=None,
    thresholds=None,
    transaction_cost_bps=None,
):
    prices = fetch_data(config['ticker'], config['start_date'], config['end_date'])
    features_df = build_feature_matrix(prices, config)
    model_config = config['model']
    walk_forward_config = config.get('walk_forward', {})

    train_windows = train_windows or walk_forward_config.get('train_windows', DEFAULT_TRAIN_WINDOWS)
    test_window_days = test_window_days or walk_forward_config.get('test_window_days', 20)
    thresholds = thresholds or walk_forward_config.get('thresholds', DEFAULT_THRESHOLDS)
    transaction_cost_bps = (
        transaction_cost_bps
        if transaction_cost_bps is not None
        else walk_forward_config.get('transaction_cost_bps', DEFAULT_TRANSACTION_COST_BPS)
    )

    rows = []
    for train_window in train_windows:
        print(f"running train-window sweep for {train_window} days")
        _, probs, dates = walk_forward_validation(
            features_df,
            model_config,
            train_window_days=train_window,
            test_window_days=test_window_days,
        )
        strategy_results = evaluate_strategies(prices, probs, dates, thresholds, transaction_cost_bps)
        results_df = strategy_results_table(
            strategy_results,
            train_window,
            test_window_days,
            avg_bullish_probability=float(np.mean(probs)),
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.extend(results_df.to_dict(orient='records'))

    results_df = pd.DataFrame(rows).sort_values('strategy_net_sharpe', ascending=False).reset_index(drop=True)
    print('\ntrain-window and threshold sweep results:')
    print(results_df.to_string(index=False))

    ensure_output_dirs()
    ticker_slug = config['ticker'].replace('.', '_')
    save_sweep_results(results_df, f"outputs/reports/train_window_sweep_{ticker_slug}.csv")
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description='Walk-forward market regime backtesting.')
    parser.add_argument('--config', default='config/config.yaml', help='Path to YAML config file')
    parser.add_argument(
        '--mode',
        choices=['single', 'sweep'],
        default='single',
        help='Run one configured walk-forward backtest or sweep multiple train windows.',
    )
    parser.add_argument('--train-window-days', type=int, help='Override the configured training window length')
    parser.add_argument('--test-window-days', type=int, help='Override the configured test window length')
    parser.add_argument(
        '--train-windows',
        type=int,
        nargs='+',
        help='Override the configured sweep train windows, e.g. --train-windows 750 1000 1250 1500',
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        help='Thresholds for threshold-based strategy, e.g. --thresholds 0.5 0.6 0.7',
    )
    parser.add_argument(
        '--transaction-cost-bps',
        type=float,
        help='Transaction cost in basis points per unit of turnover, e.g. 10 = 0.10%%',
    )
    parser.add_argument('--no-plots', action='store_true', help='Skip interactive plots for single-run mode')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.mode == 'single':
        run_single_walk_forward(
            config,
            train_window_days=args.train_window_days,
            test_window_days=args.test_window_days,
            thresholds=args.thresholds,
            transaction_cost_bps=args.transaction_cost_bps,
            show_plots=not args.no_plots,
        )
    else:
        run_train_window_sweep(
            config,
            train_windows=args.train_windows,
            test_window_days=args.test_window_days,
            thresholds=args.thresholds,
            transaction_cost_bps=args.transaction_cost_bps,
        )
