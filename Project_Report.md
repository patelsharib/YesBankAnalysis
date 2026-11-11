# Yes Bank monthly stock forecasting project

**Status: completed first modeling pass; no model is recommended for operational use.**

## Objective

Evaluate whether monthly OHLC history supports a useful next-month Yes Bank closing-price forecast. The supervised target is the next month log return. Predictions are converted to prices using the current month close. The forecast can only be evaluated as an educational historical exercise; it is not investment advice.

## Data and preparation

- 185 observations, 2005-07 through 2020-11, monthly OHLC prices.
- Missing values: 0; duplicate month labels removed: 0.
- Lagged returns, price range, opening gap, rolling volatility and 3/6 month momentum are used. Only past/current month fields predict the next month.
- After lag construction: 171 rows; first 136 rows used for fitting and time-series model selection; final 35 rows held out chronologically from 2018-01.
- No random shuffle is used. Feature scaling is contained in the Ridge pipeline.

## Model comparison

Walk-forward CV on the training period selects the candidate; the final holdout remains untouched until the final comparison. Return error is log-return absolute error. Price MAE is measured in the source price units.

| Model | Holdout return MAE | Holdout return RMSE | Holdout R² | Holdout close MAE |
|---|---:|---:|---:|---:|
| NoChangeBaseline | 0.201 | 0.273 | -0.115 | 24.24 |
| Ridge | 0.215 | 0.290 | -0.265 | 24.81 |
| RandomForest | 0.229 | 0.293 | -0.284 | 28.38 |
| HistGradientBoosting | 0.227 | 0.307 | -0.414 | 27.70 |

Walk-forward CV chose **RandomForest**: mean CV MAE 0.099, versus 0.100 for the no-change baseline. On the final holdout, however, candidate close MAE was **28.38** against **24.24** for no-change, and its directional accuracy was **40%**. Its holdout R² is below zero. The model therefore fails to demonstrate useful out-of-sample predictive value.

## Interpretation and action

- Keep the no-change forecast as the benchmark; do not use this candidate to trade, value the bank, or make customer/business decisions.
- The history is short (185 months) and spans major structural changes and extreme risk. A single chronological holdout has only 35 cases, so all scores are uncertain and regime dependent.
- A price-level model can mistake long-term scale shifts for predictive signal. Return targets and chronological validation reduce, but do not eliminate, this problem.
- If extending the study, obtain adjusted prices plus market/index, corporate action and macroeconomic data; define a decision horizon and transaction costs in advance; compare multiple rolling-origin periods and keep a final future period untouched.
- Treat feature importance as descriptive for this fitted model, not causal evidence.

## Reproduce

From this folder run `python run_project.py` (or execute the notebook). It writes 18 charts, CSV metric tables, `summary.json`, permutation importance, and a serialized experimental candidate under `outputs/`. It requires pandas, numpy, scikit-learn, matplotlib, seaborn and joblib.

## Chart guide

1. **Monthly closing price:** Monthly closing price; included to inspect trend, risk, dependence, model fit or error.
2. **OHLC price history:** OHLC price history; included to inspect trend, risk, dependence, model fit or error.
3. **Monthly log returns:** Monthly log returns; included to inspect trend, risk, dependence, model fit or error.
4. **Distribution of monthly returns:** Distribution of monthly returns; included to inspect trend, risk, dependence, model fit or error.
5. **Return spread and extreme months:** Return spread and extreme months; included to inspect trend, risk, dependence, model fit or error.
6. **Six-month rolling return volatility:** Six-month rolling return volatility; included to inspect trend, risk, dependence, model fit or error.
7. **Close-price drawdown from prior peak:** Close-price drawdown from prior peak; included to inspect trend, risk, dependence, model fit or error.
8. **Relative intramonth range vs close:** Relative intramonth range vs close; included to inspect trend, risk, dependence, model fit or error.
9. **Open-to-close return distribution:** Open-to-close return distribution; included to inspect trend, risk, dependence, model fit or error.
10. **Open vs closing price:** Open vs closing price; included to inspect trend, risk, dependence, model fit or error.
11. **Return autocorrelation by lag:** Return autocorrelation by lag; included to inspect trend, risk, dependence, model fit or error.
12. **Returns and rolling mean:** Returns and rolling mean; included to inspect trend, risk, dependence, model fit or error.
13. **Average return by calendar month:** Average return by calendar month; included to inspect trend, risk, dependence, model fit or error.
14. **Current vs prior-month return:** Current vs prior-month return; included to inspect trend, risk, dependence, model fit or error.
15. **Feature and target correlation:** Feature and target correlation; included to inspect trend, risk, dependence, model fit or error.
16. **Chronological holdout: next-month close:** Chronological holdout: next-month close; included to inspect trend, risk, dependence, model fit or error.
17. **Holdout prediction residuals:** Holdout prediction residuals; included to inspect trend, risk, dependence, model fit or error.
18. **Test-set permutation importance:** Test-set permutation importance; included to inspect trend, risk, dependence, model fit or error.
