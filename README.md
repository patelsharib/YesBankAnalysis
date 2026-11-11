# Yes Bank Monthly Stock Forecasting

An educational time-series regression project using the supplied monthly Yes Bank OHLC dataset. The analysis compares Ridge, Random Forest, and Histogram Gradient Boosting forecasts with a no-change baseline using chronological validation.

**Result:** no learned model beat the no-change baseline on the final holdout. This project does not support an operational forecasting or trading recommendation.

## Run

Install dependencies with `pip install -r requirements.txt`, then run `python run_project.py` from this directory. The notebook `Yes_Bank_Monthly_Forecasting.ipynb` runs the same script.

## Contents

- `data_YesBank_StockPrices.csv`: supplied monthly OHLC source data.
- `run_project.py`: reproducible data preparation, modeling, evaluation, and chart generation.
- `Yes_Bank_Monthly_Forecasting.ipynb`: notebook entry point and interpretation.
- `Project_Report.md`: results, limitations, and practical recommendations.
- `outputs/`: metrics, 18 charts, summary, permutation importance, and experimental model artifact.

See the report before interpreting the saved model. The artifact is for reproduction only and is not recommended for live decisions.
# YesBankAnalysis
