# Running The Walk-Forward Tools

## Prerequisites

Run these once from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Single run

Default run:

```powershell
python -m src.walk_forward --mode single
```

Best setup from the last sweep, with costs enabled:

```powershell
python -m src.walk_forward --mode single --train-window-days 1000 --thresholds 0.6 --transaction-cost-bps 10
```

Override windows, thresholds, and costs:

```powershell
python -m src.walk_forward --mode single --train-window-days 1000 --test-window-days 20 --thresholds 0.5 0.6 0.7 --transaction-cost-bps 10
```

Skip plots and just generate the reports:

```powershell
python -m src.walk_forward --mode single --no-plots
```

## Sweep run

Use the train windows, thresholds, and costs from `config/config.yaml`:

```powershell
python -m src.walk_forward --mode sweep
```

Override the sweep directly:

```powershell
python -m src.walk_forward --mode sweep --train-windows 750 1000 1250 1500 --thresholds 0.5 0.6 0.7 --test-window-days 20 --transaction-cost-bps 10
```

## Outputs

When `plot.save_figure: true`, the scripts write files into:

- `outputs/figures/`
- `outputs/reports/`

Generated artifacts include:

- walk-forward regime plot
- strategy comparison plot, net of costs
- bullish-probability plot
- per-date walk-forward CSV report with exposures, turnover, and transaction costs
- strategy comparison CSV report
- train-window sweep CSV report

## Notes

- `probability_weighted` invests a fraction of capital equal to the previous day's bullish-state probability.
- `threshold_*` strategies go fully long only when the previous day's bullish probability is at or above the threshold.
- `transaction_cost_bps` is charged per unit of daily turnover.
- RSI is included when `features.rsi_window` is set in the config.
- The sweep ranks train-window and threshold combinations by net annualized Sharpe ratio.
