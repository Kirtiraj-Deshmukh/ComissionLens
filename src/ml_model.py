"""
ml_model.py
-----------
Trains two models on the quarterly feature matrix:

  1. REGRESSION  : Predict next-quarter net alpha (continuous)
     Model: XGBoost Regressor
     Metric: RMSE, MAE, R²

  2. CLASSIFICATION : Predict whether commission is justified (binary)
     Model: XGBoost Classifier
     Metric: AUC-ROC, F1, Precision@top-decile

Temporal train/test split:
  - Train: quarters up to Q4 2022
  - Test : quarters from Q1 2023 onwards
  (No look-ahead bias: each row uses only past features)

Also saves:
  - Feature importance DataFrame
  - Test predictions (for SHAP + SIP backtest)
  - Model artifacts (.json)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, classification_report, precision_score
)
from sklearn.preprocessing import StandardScaler
import os, json, warnings
warnings.filterwarnings("ignore")

PROC_DIR   = "/home/claude/commissionlens/data/processed"
REPORT_DIR = "/home/claude/commissionlens/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURE_COLS = [
    "expense_gap", "direct_er", "regular_er",
    "aum_log", "manager_tenure_yrs", "portfolio_turnover",
    "beta_static", "rolling_sharpe_4q", "rolling_sharpe_8q",
    "rolling_beta", "information_ratio_4q", "alpha_persistence",
    "gross_alpha_q", "net_alpha_q",
    "repo_rate", "cpi", "yield_slope",
    "fii_flow_cr_qtly", "dii_flow_cr_qtly", "fii_dii_ratio",
]

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")


def load_and_split(fill_na=True):
    df = pd.read_parquet(f"{PROC_DIR}/features.parquet")
    df["quarter"] = pd.to_datetime(df["quarter"])

    if fill_na:
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(df[col].median())

    train = df[df["quarter"] <= TRAIN_CUTOFF].copy()
    test  = df[df["quarter"] >  TRAIN_CUTOFF].copy()

    X_train = train[FEATURE_COLS].values
    X_test  = test[FEATURE_COLS].values

    y_reg_train = train["net_alpha_next_q"].values
    y_reg_test  = test["net_alpha_next_q"].values

    y_cls_train = train["commission_justified"].values
    y_cls_test  = test["commission_justified"].values

    print(f"Train: {len(train)} rows ({train['quarter'].min().date()} – {train['quarter'].max().date()})")
    print(f"Test : {len(test)} rows ({test['quarter'].min().date()} – {test['quarter'].max().date()})")
    print(f"Class balance (test): {y_cls_test.mean():.2%} positive")

    return (X_train, X_test, y_reg_train, y_reg_test,
            y_cls_train, y_cls_test, train, test)


# ─────────────────────────────────────────────
# 1. Regression
# ─────────────────────────────────────────────

def train_regression(X_train, X_test, y_train, y_test):
    print("\n─── Regression: Predicting next-quarter net alpha ───")

    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"  RMSE : {rmse:.4f}  ({rmse*100:.2f}% quarterly alpha)")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    model.save_model(f"{PROC_DIR}/model_regression.json")

    # Feature importance
    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance_reg": model.feature_importances_,
    }).sort_values("importance_reg", ascending=False)

    return model, preds, fi


# ─────────────────────────────────────────────
# 2. Classification
# ─────────────────────────────────────────────

def train_classification(X_train, X_test, y_train, y_test):
    print("\n─── Classification: Predicting commission justification ───")

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1  = f1_score(y_test, preds)

    # Precision at top decile (highest predicted probability)
    top_decile_n = max(1, len(proba) // 10)
    top_idx      = np.argsort(proba)[::-1][:top_decile_n]
    prec_top10   = y_test[top_idx].mean()

    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Precision@Top10% : {prec_top10:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds,
                                target_names=["Not Justified", "Justified"]))

    model.save_model(f"{PROC_DIR}/model_classification.json")

    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance_cls": model.feature_importances_,
    }).sort_values("importance_cls", ascending=False)

    return model, proba, preds, fi


# ─────────────────────────────────────────────
# 3. Feature Importance Report
# ─────────────────────────────────────────────

def build_importance_report(fi_reg, fi_cls):
    fi = fi_reg.merge(fi_cls, on="feature")
    fi["importance_mean"] = (fi["importance_reg"] + fi["importance_cls"]) / 2
    fi = fi.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    fi.to_csv(f"{REPORT_DIR}/feature_importance.csv", index=False)
    print("\n─── Top 10 Features (avg regression + classification) ───")
    print(fi[["feature", "importance_reg", "importance_cls", "importance_mean"]].head(10).to_string(index=False))
    return fi


# ─────────────────────────────────────────────
# 4. Save Test Predictions
# ─────────────────────────────────────────────

def save_test_predictions(test_df, reg_preds, cls_proba, cls_preds):
    test_df = test_df.copy()
    test_df["pred_net_alpha"]           = reg_preds
    test_df["pred_justified_proba"]     = cls_proba
    test_df["pred_commission_justified"] = cls_preds
    out = f"{PROC_DIR}/test_predictions.parquet"
    test_df.to_parquet(out, index=False)
    print(f"\n✅ Test predictions saved → {out}")
    return test_df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("CommissionLens — ML Modeling")
    print("=" * 55)

    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test,
     train_df, test_df) = load_and_split()

    reg_model,  reg_preds,              fi_reg = train_regression(
        X_train, X_test, y_reg_train, y_reg_test)

    cls_model, cls_proba, cls_preds, fi_cls = train_classification(
        X_train, X_test, y_cls_train, y_cls_test)

    fi_combined = build_importance_report(fi_reg, fi_cls)
    pred_df = save_test_predictions(test_df, reg_preds, cls_proba, cls_preds)

    # Summary metrics dict
    metrics = {
        "regression": {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_preds))),
            "mae":  float(mean_absolute_error(y_reg_test, reg_preds)),
            "r2":   float(r2_score(y_reg_test, reg_preds)),
        },
        "classification": {
            "auc_roc": float(roc_auc_score(y_cls_test, cls_proba)),
            "f1":      float(f1_score(y_cls_test, cls_preds)),
            "prec_top10": float(
                y_cls_test[np.argsort(cls_proba)[::-1][:max(1, len(cls_proba)//10)]].mean()
            ),
        }
    }
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved → {REPORT_DIR}/metrics.json")
    return reg_model, cls_model, pred_df, fi_combined


if __name__ == "__main__":
    main()
