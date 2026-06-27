"""
walk_forward_cv.py
------------------
Generates out-of-sample predictions for ALL quarters 2019–2023
using expanding-window walk-forward cross-validation.

For each quarter Q:
  - Train on all data up to Q-1
  - Predict on Q
  - Collect predictions

This ensures the SIP backtest has genuine OOS predictions throughout.
Also computes per-period metrics to show model calibration over time.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
import warnings, os
warnings.filterwarnings("ignore")

PROC_DIR = "/home/claude/commissionlens/data/processed"
RAW_DIR  = "/home/claude/commissionlens/data/raw"

FEATURE_COLS = [
    "expense_gap", "direct_er", "regular_er",
    "aum_log", "manager_tenure_yrs", "portfolio_turnover",
    "beta_static", "rolling_sharpe_4q", "rolling_sharpe_8q",
    "rolling_beta", "information_ratio_4q", "alpha_persistence",
    "gross_alpha_q", "net_alpha_q",
    "repo_rate", "cpi", "yield_slope",
    "fii_flow_cr_qtly", "dii_flow_cr_qtly", "fii_dii_ratio",
]

MIN_TRAIN_QUARTERS = 4   # Need at least 4 quarters before first prediction


def run_walk_forward():
    print("=" * 55)
    print("CommissionLens — Walk-Forward Cross-Validation")
    print("=" * 55)

    df = pd.read_parquet(f"{PROC_DIR}/features.parquet")
    df["quarter"] = pd.to_datetime(df["quarter"])
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    quarters = sorted(df["quarter"].unique())
    print(f"Total quarters: {len(quarters)} ({str(quarters[0])[:10]} → {str(quarters[-1])[:10]})")

    all_preds = []
    metrics_log = []

    cls_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    reg_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    for i, q in enumerate(quarters):
        if i < MIN_TRAIN_QUARTERS:
            continue   # not enough history to train

        train = df[df["quarter"] < q].copy()
        test  = df[df["quarter"] == q].copy()

        if len(train) < 50 or len(test) == 0:
            continue

        X_train = train[FEATURE_COLS].values
        X_test  = test[FEATURE_COLS].values
        y_reg_train = train["net_alpha_next_q"].values
        y_cls_train = train["commission_justified"].values
        y_reg_test  = test["net_alpha_next_q"].values
        y_cls_test  = test["commission_justified"].values

        # Classification
        scale_pos = max(0.1, (y_cls_train == 0).sum() / max(1, (y_cls_train == 1).sum()))
        cls_model = xgb.XGBClassifier(scale_pos_weight=scale_pos, **cls_params)
        cls_model.fit(X_train, y_cls_train, verbose=False)
        cls_proba = cls_model.predict_proba(X_test)[:, 1]
        cls_preds = (cls_proba >= 0.5).astype(int)

        # Regression
        reg_model = xgb.XGBRegressor(**reg_params)
        reg_model.fit(X_train, y_reg_train, verbose=False)
        reg_preds = reg_model.predict(X_test)

        # Metrics
        try:
            auc = roc_auc_score(y_cls_test, cls_proba)
            f1  = f1_score(y_cls_test, cls_preds, zero_division=0)
            rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
        except Exception:
            auc, f1, rmse = np.nan, np.nan, np.nan

        metrics_log.append({
            "quarter": q, "auc": auc, "f1": f1, "rmse": rmse,
            "n_train": len(train), "n_test": len(test),
        })

        # Store predictions
        test = test.copy()
        test["pred_net_alpha"]            = reg_preds
        test["pred_justified_proba"]      = cls_proba
        test["pred_commission_justified"] = cls_preds
        all_preds.append(test)

        if i % 4 == 0:
            print(f"  Q {str(q)[:10]}  train={len(train):4d}  AUC={auc:.3f}  F1={f1:.3f}  RMSE={rmse:.4f}")

    # Combine
    pred_df  = pd.concat(all_preds, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_log)

    out_pred = f"{PROC_DIR}/walkforward_predictions.parquet"
    out_metrics = f"{PROC_DIR}/walkforward_metrics.csv"
    pred_df.to_parquet(out_pred, index=False)
    metrics_df.to_csv(out_metrics, index=False)

    print(f"\n✅ Walk-forward predictions: {len(pred_df):,} rows → {out_pred}")
    print(f"✅ Metrics log              : {len(metrics_df)} quarters → {out_metrics}")
    print(f"\nMean AUC across quarters : {metrics_df['auc'].mean():.4f}")
    print(f"Mean F1  across quarters : {metrics_df['f1'].mean():.4f}")
    print(f"Mean RMSE                : {metrics_df['rmse'].mean():.4f}")

    return pred_df, metrics_df


if __name__ == "__main__":
    run_walk_forward()
