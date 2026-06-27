"""
feature_engineering.py
-----------------------
Builds a quarterly fund-level feature matrix from raw data.

Features engineered:
  Fund-level:
    - expense_gap            : regular_er - direct_er
    - rolling_sharpe_4q      : 4-quarter rolling Sharpe ratio (direct NAV)
    - rolling_sharpe_8q      : 8-quarter rolling Sharpe ratio
    - beta                   : systematic risk vs Nifty 50
    - information_ratio_4q   : alpha / tracking error (4Q rolling)
    - aum_log                : log(AUM in crores)
    - manager_tenure_yrs
    - portfolio_turnover
    - alpha_persistence      : AR(1) coefficient of quarterly alpha series

  Macro (quarter-end values):
    - repo_rate
    - cpi
    - yield_slope
    - fii_flow_cr_qtly       : summed FII flow for the quarter
    - dii_flow_cr_qtly       : summed DII flow for the quarter
    - fii_dii_ratio          : FII / (FII + DII)

  Target:
    - net_alpha_next_q       : next quarter's net alpha (gross alpha - expense_gap/4)
    - commission_justified   : 1 if net_alpha_next_q > 0 else 0
"""

import numpy as np
import pandas as pd
import os

RAW_DIR  = "/home/claude/commissionlens/data/raw"
PROC_DIR = "/home/claude/commissionlens/data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def quarterly_returns(nav_series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Convert daily NAV to quarterly returns."""
    df = pd.DataFrame({"nav": nav_series.values}, index=dates)
    qret = df["nav"].resample("QE").last().pct_change()
    return qret


def rolling_sharpe(qreturns: pd.Series, window: int, rf_quarterly: float = 0.015) -> pd.Series:
    """Rolling Sharpe ratio over `window` quarters (annualised)."""
    excess = qreturns - rf_quarterly
    sharpe = excess.rolling(window).mean() / (excess.rolling(window).std() + 1e-9)
    return sharpe * np.sqrt(4)   # annualise


def compute_beta(fund_qret: pd.Series, bench_qret: pd.Series, window: int = 8) -> pd.Series:
    """Rolling beta over `window` quarters."""
    betas = []
    idx = []
    for i in range(window, len(fund_qret) + 1):
        f = fund_qret.iloc[i - window:i]
        b = bench_qret.iloc[i - window:i]
        valid = f.notna() & b.notna()
        if valid.sum() < 4:
            betas.append(np.nan)
        else:
            cov = np.cov(f[valid], b[valid])
            betas.append(cov[0, 1] / (cov[1, 1] + 1e-12))
        idx.append(fund_qret.index[i - 1])
    return pd.Series(betas, index=idx)


def compute_information_ratio(alpha_series: pd.Series, window: int = 4) -> pd.Series:
    """IR = mean(alpha) / std(alpha) over rolling window."""
    return (alpha_series.rolling(window).mean() /
            (alpha_series.rolling(window).std() + 1e-9) * np.sqrt(4))


def alpha_persistence(alpha_series: pd.Series, window: int = 8) -> pd.Series:
    """AR(1) coefficient of alpha (how much past alpha predicts future alpha)."""
    persist = []
    idx = []
    for i in range(window, len(alpha_series) + 1):
        a = alpha_series.iloc[i - window:i].dropna()
        if len(a) < 4:
            persist.append(np.nan)
        else:
            corr = np.corrcoef(a[:-1].values, a[1:].values)
            persist.append(corr[0, 1])
        idx.append(alpha_series.index[i - 1])
    return pd.Series(persist, index=idx)


# ─────────────────────────────────────────────
# Main Feature Builder
# ─────────────────────────────────────────────

def build_features():
    print("Loading raw data...")
    fund_pairs = pd.read_csv(f"{RAW_DIR}/fund_pairs.csv")
    benchmark  = pd.read_csv(f"{RAW_DIR}/benchmark_nifty50.csv", parse_dates=["date"])
    macro      = pd.read_csv(f"{RAW_DIR}/macro_data.csv", parse_dates=["date"])
    nav_df     = pd.read_parquet(f"{RAW_DIR}/nav_histories.parquet")
    nav_df["date"] = pd.to_datetime(nav_df["date"])

    # Benchmark quarterly returns
    bench_qret = (
        benchmark.set_index("date")["nifty50_close"]
        .resample("QE").last()
        .pct_change()
    )

    # Macro → quarterly aggregates
    macro = macro.set_index("date")
    macro_q = macro.resample("QE").agg({
        "repo_rate":   "last",
        "cpi":         "mean",
        "yield_slope": "last",
        "fii_flow_cr": "sum",
        "dii_flow_cr": "sum",
    }).rename(columns={"fii_flow_cr": "fii_flow_cr_qtly",
                       "dii_flow_cr": "dii_flow_cr_qtly"})
    macro_q["fii_dii_ratio"] = (
        macro_q["fii_flow_cr_qtly"] /
        (macro_q["fii_flow_cr_qtly"].abs() + macro_q["dii_flow_cr_qtly"].abs() + 1)
    )

    all_features = []
    total = len(fund_pairs)

    print(f"Engineering features for {total} funds...")

    for idx_f, fund in fund_pairs.iterrows():
        fid = fund["fund_id"]

        nav_fund = nav_df[nav_df["fund_id"] == fid].set_index("date").sort_index()
        if nav_fund.empty:
            continue

        dates  = nav_fund.index
        nav_dir = nav_fund["nav_direct"]
        nav_reg = nav_fund["nav_regular"]

        # Quarterly NAVs
        nav_dir_q = nav_dir.resample("QE").last()
        nav_reg_q = nav_reg.resample("QE").last()

        dir_qret = nav_dir_q.pct_change()
        reg_qret = nav_reg_q.pct_change()

        # Align benchmark
        bench_aligned = bench_qret.reindex(dir_qret.index)

        # Quarterly gross alpha (direct return - benchmark return)
        gross_alpha_q = dir_qret - bench_aligned

        # Net alpha = gross alpha - expense_gap/4 (quarterly cost)
        expense_gap_q = fund["expense_gap"] / 100 / 4
        net_alpha_q   = gross_alpha_q - expense_gap_q

        # Features
        sh4  = rolling_sharpe(dir_qret, window=4)
        sh8  = rolling_sharpe(dir_qret, window=8)
        beta_rolling = compute_beta(dir_qret, bench_aligned, window=8)
        ir4  = compute_information_ratio(gross_alpha_q, window=4)
        ap   = alpha_persistence(gross_alpha_q, window=8)

        quarters = dir_qret.index[1:]   # skip first NaN quarter
        for q in quarters:
            if q not in net_alpha_q.index:
                continue
            # Target: NEXT quarter's net alpha
            q_idx = net_alpha_q.index.get_loc(q)
            if q_idx + 1 >= len(net_alpha_q):
                continue
            next_q = net_alpha_q.index[q_idx + 1]
            target_net_alpha = net_alpha_q.iloc[q_idx + 1]

            row = {
                "fund_id":           fid,
                "quarter":           q,
                # Fund-level features
                "expense_gap":       fund["expense_gap"],
                "direct_er":         fund["direct_er"],
                "regular_er":        fund["regular_er"],
                "aum_log":           np.log1p(fund["aum_cr"]),
                "manager_tenure_yrs": fund["manager_tenure_yrs"],
                "portfolio_turnover": fund["portfolio_turnover"],
                "beta_static":       fund["beta"],
                "rolling_sharpe_4q": sh4.get(q, np.nan),
                "rolling_sharpe_8q": sh8.get(q, np.nan),
                "rolling_beta":      beta_rolling.get(q, np.nan),
                "information_ratio_4q": ir4.get(q, np.nan),
                "alpha_persistence": ap.get(q, np.nan),
                "gross_alpha_q":     gross_alpha_q.get(q, np.nan),
                "net_alpha_q":       net_alpha_q.get(q, np.nan),
                # Macro features
                **{k: macro_q.loc[q, k] if q in macro_q.index else np.nan
                   for k in macro_q.columns},
                # Targets
                "net_alpha_next_q":      target_net_alpha,
                "commission_justified":  int(target_net_alpha > 0),
            }
            all_features.append(row)

        if (idx_f + 1) % 50 == 0:
            print(f"  Processed {idx_f + 1}/{total} funds...")

    features_df = pd.DataFrame(all_features)
    features_df = features_df.dropna(subset=["net_alpha_next_q"])
    features_df["quarter"] = pd.to_datetime(features_df["quarter"])
    features_df = features_df.sort_values(["quarter", "fund_id"]).reset_index(drop=True)

    out_path = f"{PROC_DIR}/features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"\n✅ Feature matrix saved → {out_path}")
    print(f"   Shape  : {features_df.shape}")
    print(f"   Quarters: {features_df['quarter'].nunique()}")
    print(f"   Funds   : {features_df['fund_id'].nunique()}")
    print(f"\nClass balance (commission_justified):")
    print(features_df["commission_justified"].value_counts(normalize=True).round(3))
    print("\nSample rows:")
    print(features_df.head(3).T)

    return features_df


if __name__ == "__main__":
    build_features()
