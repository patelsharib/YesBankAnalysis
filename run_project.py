"""Reproducible Yes Bank monthly stock return forecasting project.

Run from this directory with: python run_project.py
The model uses only information available at the end of month t to predict
the close-to-close return for month t+1. This is an analysis,
not investment advice.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_YesBank_StockPrices.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def load_data():
    """Load and validate the supplied monthly OHLC data."""
    if not DATA.exists():
        raise FileNotFoundError(f"Expected input data at {DATA}")
    raw = pd.read_csv(DATA)
    required = {"Date", "Open", "High", "Low", "Close"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(raw.columns))}")
    df = raw.copy()
    # Month labels use two-digit years; explicitly map 00–68 to 2000s,
    # and 69–99 to 1900s (all observed records are 2005–2020).
    df["Date"] = pd.to_datetime(df["Date"], format="%b-%y")
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values; review the source before modeling.")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive to compute log returns.")
    return df


def make_features(df):
    """Construct lagged, scale-stable features and a next-month return label."""
    d = df.copy()
    d["ret"] = np.log(d["Close"]).diff()
    d["intramonth_oc"] = np.log(d["Close"] / d["Open"])
    d["range_pct"] = (d["High"] - d["Low"]) / d["Close"]
    d["gap"] = np.log(d["Open"] / d["Close"].shift(1))
    for lag in (1, 2, 3, 6, 12):
        d[f"ret_lag_{lag}"] = d["ret"].shift(lag)
    d["vol_3"] = d["ret"].rolling(3).std()
    d["vol_6"] = d["ret"].rolling(6).std()
    d["mom_3"] = np.log(d["Close"] / d["Close"].shift(3))
    d["mom_6"] = np.log(d["Close"] / d["Close"].shift(6))
    d["target_return"] = d["ret"].shift(-1)
    d["target_date"] = d["Date"].shift(-1)
    d["target_close"] = d["Close"].shift(-1)
    features = ["ret", "intramonth_oc", "range_pct", "gap", "ret_lag_1", "ret_lag_2",
                "ret_lag_3", "ret_lag_6", "ret_lag_12", "vol_3", "vol_6", "mom_3", "mom_6"]
    frame = d.dropna(subset=features + ["target_return"]).reset_index(drop=True)
    return d, frame, features


def metric_row(name, y, pred):
    return {"model": name, "MAE_return": mean_absolute_error(y, pred),
            "RMSE_return": mean_squared_error(y, pred) ** 0.5,
            "R2_return": r2_score(y, pred)}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close()


def make_charts(df, d, frame, features, split, y_test, preds, best_name, importance):
    """Save 18 exploratory and diagnostic charts with chart-specific titles."""
    # 1. Long-run close path: establishes scale and major regimes.
    plt.figure(figsize=(11, 4)); plt.plot(df.Date, df.Close); plt.title("1. Monthly closing price"); plt.ylabel("Price"); savefig("01_close_series.png")
    # 2. OHLC levels compare within-month price dimensions.
    plt.figure(figsize=(11, 4));
    for c in ["Open", "High", "Low", "Close"]: plt.plot(df.Date, df[c], label=c, alpha=.8)
    plt.legend(ncol=4); plt.title("2. OHLC price history"); plt.ylabel("Price"); savefig("02_ohlc.png")
    # 3. Return path highlights high-volatility episodes.
    plt.figure(figsize=(11, 4)); plt.plot(d.Date, d.ret * 100); plt.axhline(0,color="black",lw=.8); plt.title("3. Monthly log returns"); plt.ylabel("Percent"); savefig("03_returns.png")
    # 4. Histogram describes return distribution and tails.
    plt.figure(figsize=(7, 4)); sns.histplot(d.ret.dropna()*100, bins=25, kde=True); plt.title("4. Distribution of monthly returns"); plt.xlabel("Log return (%)"); savefig("04_return_distribution.png")
    # 5. Box plot gives robust spread/outlier view.
    plt.figure(figsize=(7, 3)); sns.boxplot(x=d.ret*100); plt.title("5. Return spread and extreme months"); plt.xlabel("Log return (%)"); savefig("05_return_box.png")
    # 6. Rolling vol captures changing risk.
    plt.figure(figsize=(11, 4)); plt.plot(d.Date, d.ret.rolling(6).std()*100); plt.title("6. Six-month rolling return volatility"); plt.ylabel("Std. dev. (%)"); savefig("06_rolling_volatility.png")
    # 7. Drawdown makes peak-to-trough losses visible.
    wealth=df.Close/df.Close.iloc[0]; dd=wealth/wealth.cummax()-1
    plt.figure(figsize=(11, 4)); plt.fill_between(df.Date, dd*100, 0, alpha=.45); plt.title("7. Close-price drawdown from prior peak"); plt.ylabel("Drawdown (%)"); savefig("07_drawdown.png")
    # 8. Monthly range against close reveals absolute range growth with price.
    plt.figure(figsize=(7, 4)); plt.scatter(df.Close,(df.High-df.Low)/df.Close,alpha=.65); plt.title("8. Relative intramonth range vs close"); plt.xlabel("Close"); plt.ylabel("(High − Low) / Close"); savefig("08_range_vs_price.png")
    # 9. Open-close change distribution checks directional intramonth bias.
    plt.figure(figsize=(7, 4)); sns.histplot(d.intramonth_oc.dropna()*100,bins=25,kde=True); plt.axvline(0,color="black"); plt.title("9. Open-to-close return distribution"); plt.xlabel("Log return (%)"); savefig("09_open_close.png")
    # 10. Open-close relationship shows prices move together over regimes.
    plt.figure(figsize=(6, 5)); plt.scatter(df.Open,df.Close,alpha=.65); lim=max(df.Open.max(),df.Close.max()); plt.plot([0,lim],[0,lim],"--",color="gray"); plt.title("10. Open vs closing price"); plt.xlabel("Open"); plt.ylabel("Close"); savefig("10_open_close_scatter.png")
    # 11. Return autocorrelation checks whether simple momentum is apparent.
    lags=range(1,13); ac=[d.ret.autocorr(lag=i) for i in lags]
    plt.figure(figsize=(8, 4)); plt.bar(list(lags),ac); plt.axhline(0,color="black"); plt.title("11. Return autocorrelation by lag"); plt.xlabel("Lag (months)"); savefig("11_autocorrelation.png")
    # 12. Rolling mean compares short-term trend to noise.
    plt.figure(figsize=(11, 4)); plt.plot(d.Date,d.ret*100,alpha=.35,label="Monthly"); plt.plot(d.Date,d.ret.rolling(6).mean()*100,label="6-month mean"); plt.axhline(0,color="black",lw=.7); plt.legend(); plt.title("12. Returns and rolling mean"); savefig("12_rolling_mean.png")
    # 13. Calendar month seasonality is descriptive only.
    monthly=d.assign(month=d.Date.dt.month).groupby("month").ret.mean()*100
    plt.figure(figsize=(8, 4)); plt.bar(monthly.index,monthly.values); plt.axhline(0,color="black"); plt.title("13. Average return by calendar month"); plt.xlabel("Calendar month"); plt.ylabel("Mean log return (%)"); savefig("13_calendar_month.png")
    # 14. Lag one relationship tests linear predictability visually.
    plt.figure(figsize=(6, 5)); plt.scatter(d.ret.shift(1)*100,d.ret*100,alpha=.65); plt.axhline(0,color="gray",lw=.7); plt.axvline(0,color="gray",lw=.7); plt.title("14. Current vs prior-month return"); plt.xlabel("Prior-month return (%)"); plt.ylabel("Current return (%)"); savefig("14_lag_scatter.png")
    # 15. Correlation heatmap summarizes feature relationships.
    plt.figure(figsize=(10, 8)); sns.heatmap(frame[features+['target_return']].corr(),cmap="vlag",center=0); plt.title("15. Feature and target correlation"); savefig("15_correlation.png")
    # 16. Holdout predictions test whether the selected model beats baseline.
    idx=frame.index[split:]; dates=frame.loc[idx,"target_date"]
    actual=np.expm1(y_test); baseline=np.zeros(len(y_test)); model_return=preds[best_name]
    actual_close=frame.loc[idx,"Close"].to_numpy()*np.exp(y_test.to_numpy())
    pred_close=frame.loc[idx,"Close"].to_numpy()*np.exp(model_return)
    naive_close=frame.loc[idx,"Close"].to_numpy()
    plt.figure(figsize=(11, 4)); plt.plot(dates,actual_close,label="Actual"); plt.plot(dates,pred_close,label=best_name); plt.plot(dates,naive_close,label="No-change baseline",alpha=.7); plt.legend(); plt.title("16. Chronological holdout: next-month close"); plt.ylabel("Price"); savefig("16_holdout_close.png")
    # 17. Residuals reveal bias and large misses.
    resid=y_test.to_numpy()-model_return
    plt.figure(figsize=(8, 4)); plt.scatter(dates,resid*100); plt.axhline(0,color="black"); plt.title("17. Holdout prediction residuals"); plt.ylabel("Actual − predicted return (%)"); savefig("17_residuals.png")
    # 18. Permutation importance is computed on untouched test rows.
    plt.figure(figsize=(8, 5)); top=importance.sort_values().tail(10); plt.barh(top.index,top.values); plt.title("18. Test-set permutation importance (RMSE increase)"); savefig("18_importance.png")
    return actual_close, pred_close, naive_close


def main():
    df=load_data()
    d, frame, features=make_features(df)
    # Reserve the last 20% of supervised observations as a final chronological holdout.
    split=int(len(frame)*.80)
    train=frame.iloc[:split]; test=frame.iloc[split:]
    X_train,y_train=train[features],train.target_return
    X_test,y_test=test[features],test.target_return
    models={
        "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
        "RandomForest":RandomForestRegressor(n_estimators=400,min_samples_leaf=4,max_features=.8,random_state=42,n_jobs=-1),
        "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=2.0,random_state=42),
    }
    # Compare candidates using expanding-window cross-validation on training history only.
    tscv=TimeSeriesSplit(n_splits=5)
    cv_rows=[]
    for name,model in models.items():
        scores=-cross_val_score(model,X_train,y_train,cv=tscv,scoring="neg_mean_absolute_error")
        cv_rows.append({"model":name,"walk_forward_cv_MAE_return":scores.mean(),"cv_std":scores.std()})
    # Zero predicted return corresponds to the transparent no-change close baseline.
    cv_rows.append({"model":"NoChangeBaseline","walk_forward_cv_MAE_return":mean_absolute_error(y_train,np.zeros(len(y_train))),"cv_std":np.nan})
    cv=pd.DataFrame(cv_rows).sort_values("walk_forward_cv_MAE_return")
    scores={"NoChangeBaseline":np.zeros(len(y_test))}
    for name,model in models.items():
        model.fit(X_train,y_train)
        scores[name]=model.predict(X_test)
    results=pd.DataFrame([metric_row(name,y_test,p) for name,p in scores.items()])
    # Compare next-month forecast with the actual next-month close, never the
    # current close. This keeps the price-scale metric aligned to the target.
    results["holdout_close_MAE"]= [mean_absolute_error(test.target_close,test.Close*np.exp(p)) for p in scores.values()]
    # Select using training-only walk-forward CV, with baseline included.
    best_name=cv.iloc[0].model
    if best_name=="NoChangeBaseline":
        final_model=None
        importance=pd.Series(0.0,index=features)
    else:
        final_model=models[best_name]
        perm=permutation_importance(final_model,X_test,y_test,n_repeats=30,random_state=42,scoring="neg_root_mean_squared_error")
        importance=pd.Series(perm.importances_mean,index=features)
    actual_close,pred_close,naive_close=make_charts(df,d,frame,features,split,y_test,scores,best_name,importance)
    results.to_csv(OUT/"holdout_metrics.csv",index=False)
    cv.to_csv(OUT/"walk_forward_cv.csv",index=False)
    importance.sort_values(ascending=False).to_csv(OUT/"permutation_importance.csv",header=["rmse_increase"])
    if final_model is not None:
        joblib.dump({"model":final_model,"features":features,"target":"next_month_log_return","last_training_date":str(train.target_date.iloc[-1])},OUT/"yesbank_return_model.joblib")
    summary={"rows":len(df),"first_month":str(df.Date.min().date()),"last_month":str(df.Date.max().date()),"duplicates_removed":int(pd.read_csv(DATA).Date.duplicated().sum()),"missing_values":int(df.isna().sum().sum()),"supervised_rows":len(frame),"train_rows":len(train),"holdout_rows":len(test),"holdout_start":str(test.target_date.iloc[0].date()),"selected_model_by_walk_forward_cv":best_name,"test_close_mae_model":float(mean_absolute_error(test.target_close,test.Close*np.exp(scores[best_name]))),"test_close_mae_no_change":float(mean_absolute_error(test.target_close,test.Close)),"test_directional_accuracy":float(np.mean(np.sign(y_test.to_numpy())==np.sign(scores[best_name]))),"mean_monthly_log_return":float(d.ret.mean()),"median_monthly_log_return":float(d.ret.median()),"max_drawdown":float((df.Close/df.Close.cummax()-1).min())}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("\nHoldout metrics:\n",results.to_string(index=False))
    print("\nWalk-forward CV:\n",cv.to_string(index=False))
    print(f"\nSaved 18 charts and model artifacts in {OUT}")


if __name__ == "__main__":
    main()
