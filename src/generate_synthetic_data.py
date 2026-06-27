"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic data for Indian equity mutual funds.
All statistical properties mirror real Indian MF market characteristics:
  - NAV histories from 2019-2024 for 250 equity fund pairs
  - Expense ratio gaps (regular - direct) between 0.5% and 1.5%
  - Nifty 50 benchmark returns (calibrated to actual 2019-2024 performance)
  - RBI macro data (repo rate, CPI, yield curve slope, FII/DII flows)
  - Fund metadata (AUM, manager tenure, portfolio turnover, Sharpe, beta)

Replace with real API calls (mfapi.in, AMFI, RBI DBIE) when running locally.
See fetch_real_data.py for the real data fetching logic.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

RAW_DIR = "/home/claude/commissionlens/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Fund Pairs (AMFI-style)
# ─────────────────────────────────────────────

AMC_LIST = [
    "SBI Mutual Fund", "HDFC Mutual Fund", "ICICI Prudential Mutual Fund",
    "Nippon India Mutual Fund", "Kotak Mahindra Mutual Fund", "Axis Mutual Fund",
    "UTI Mutual Fund", "Mirae Asset Mutual Fund", "DSP Mutual Fund",
    "Franklin Templeton Mutual Fund", "Aditya Birla Sun Life Mutual Fund",
    "Tata Mutual Fund", "PGIM India Mutual Fund", "Edelweiss Mutual Fund",
    "Bandhan Mutual Fund"
]

SUBCATEGORIES = [
    "Large Cap Fund", "Mid Cap Fund", "Small Cap Fund", "Flexi Cap Fund",
    "ELSS", "Focused Fund", "Multi Cap Fund", "Large & Mid Cap Fund",
    "Value Fund", "Contra Fund"
]

def generate_fund_pairs(n=250):
    """Generate a DataFrame of regular/direct fund pairs."""
    records = []
    scheme_code = 100000
    for i in range(n):
        amc = AMC_LIST[i % len(AMC_LIST)]
        subcat = SUBCATEGORIES[i % len(SUBCATEGORIES)]
        base_name = f"{amc.split()[0]} {subcat} {i+1}"
        expense_gap = np.round(np.random.uniform(0.5, 1.5), 2)   # regular - direct
        direct_er   = np.round(np.random.uniform(0.3, 0.8), 2)
        regular_er  = np.round(direct_er + expense_gap, 2)
        aum_cr      = np.round(np.random.lognormal(mean=7, sigma=1.5), 0)  # in crores
        manager_tenure_yrs = np.round(np.random.uniform(1, 15), 1)
        portfolio_turnover = np.round(np.random.uniform(0.2, 2.5), 2)
        beta        = np.round(np.random.uniform(0.7, 1.3), 3)
        records.append({
            "fund_id":           i,
            "amc":               amc,
            "subcategory":       subcat,
            "base_name":         base_name,
            "regular_code":      scheme_code,
            "direct_code":       scheme_code + 1,
            "direct_er":         direct_er,
            "regular_er":        regular_er,
            "expense_gap":       expense_gap,
            "aum_cr":            aum_cr,
            "manager_tenure_yrs": manager_tenure_yrs,
            "portfolio_turnover": portfolio_turnover,
            "beta":              beta,
        })
        scheme_code += 2
    df = pd.DataFrame(records)
    path = f"{RAW_DIR}/fund_pairs.csv"
    df.to_csv(path, index=False)
    print(f"[fund_pairs] {len(df)} pairs saved → {path}")
    return df


# ─────────────────────────────────────────────
# 2. Benchmark (Nifty 50) Daily Returns
# ─────────────────────────────────────────────

def generate_benchmark(start="2018-01-01", end="2024-06-30"):
    """
    Simulate Nifty 50 daily prices calibrated to real 2019-2024 trajectory:
      - CAGR ~13%, annual vol ~18%
      - COVID crash (Feb-Mar 2020), sharp recovery, 2022 correction
    """
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Base GBM
    mu_daily    = 0.13 / 252
    sigma_daily = 0.18 / np.sqrt(252)
    log_returns = np.random.normal(mu_daily - 0.5 * sigma_daily**2, sigma_daily, n)

    # COVID crash: Feb 19 – Mar 23 2020
    covid_mask = (dates >= "2020-02-19") & (dates <= "2020-03-23")
    log_returns[covid_mask] += np.random.normal(-0.025, 0.015, covid_mask.sum())

    # COVID recovery: Apr–Dec 2020
    recovery_mask = (dates >= "2020-04-01") & (dates <= "2020-12-31")
    log_returns[recovery_mask] += 0.004

    # 2022 correction: Jan–Jun 2022
    corr_mask = (dates >= "2022-01-01") & (dates <= "2022-06-30")
    log_returns[corr_mask] -= 0.003

    prices = 10000 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({"date": dates, "nifty50_close": np.round(prices, 2)})
    df["nifty50_return"] = df["nifty50_close"].pct_change()

    path = f"{RAW_DIR}/benchmark_nifty50.csv"
    df.to_csv(path, index=False)
    print(f"[benchmark] {len(df)} trading days saved → {path}")
    return df


# ─────────────────────────────────────────────
# 3. NAV Histories for All Funds
# ─────────────────────────────────────────────

def generate_nav_histories(fund_pairs, benchmark):
    """
    Each fund's daily NAV is driven by:
      - Benchmark return (weighted by beta)
      - Fund-specific alpha (varies quarterly)
      - Expense ratio drag (compounded daily)
      - Idiosyncratic noise
    Direct plan has lower expense ratio → higher NAV over time.
    """
    bench = benchmark.set_index("date")["nifty50_return"].fillna(0)
    dates = bench.index
    all_records = []

    print(f"[nav] Generating NAV histories for {len(fund_pairs)} fund pairs...")

    for _, fund in fund_pairs.iterrows():
        fid  = fund["fund_id"]
        beta = fund["beta"]

        # Quarter-varying alpha signal (annualised), mean-reverting
        quarters = pd.period_range(dates.min(), dates.max(), freq="Q")
        alpha_q   = {}
        prev = np.random.normal(0.01, 0.03)
        for q in quarters:
            prev = 0.5 * prev + 0.5 * np.random.normal(0.005, 0.04)
            alpha_q[q] = prev

        nav_reg = 10.0
        nav_dir = 10.0
        daily_reg_drag = fund["regular_er"] / 100 / 252
        daily_dir_drag = fund["direct_er"]  / 100 / 252

        for dt in dates:
            bench_ret = bench.loc[dt]
            q = pd.Period(dt, freq="Q")
            alpha_ann = alpha_q.get(q, 0)
            alpha_daily = alpha_ann / 252
            idio = np.random.normal(0, 0.006)

            gross_ret = beta * bench_ret + alpha_daily + idio

            nav_reg *= (1 + gross_ret - daily_reg_drag)
            nav_dir *= (1 + gross_ret - daily_dir_drag)

            all_records.append({
                "fund_id":      fid,
                "date":         dt,
                "nav_regular":  round(nav_reg, 4),
                "nav_direct":   round(nav_dir, 4),
            })

    df = pd.DataFrame(all_records)
    path = f"{RAW_DIR}/nav_histories.parquet"
    df.to_parquet(path, index=False)
    print(f"[nav] {len(df):,} rows saved → {path}")
    return df


# ─────────────────────────────────────────────
# 4. Macro Data (RBI DBIE-style)
# ─────────────────────────────────────────────

def generate_macro_data(start="2018-01-01", end="2024-06-30"):
    """
    Monthly macro variables calibrated to Indian data 2019-2024:
      - Repo rate: 6.5% → 4% (COVID cuts) → 6.5% (2023 hikes)
      - CPI: 4-7% range
      - Yield curve slope: 10Y - 3M G-sec spread
      - FII net flows (₹ crore), DII net flows
    """
    months = pd.date_range(start, end, freq="MS")
    n = len(months)
    records = []

    repo = 6.5
    cpi  = 4.5
    slope = 1.2
    fii_flow = 0.0
    dii_flow = 5000.0

    for i, dt in enumerate(months):
        # Repo rate trajectory
        if dt < pd.Timestamp("2020-03-01"):
            repo = max(5.75, repo - np.random.choice([-0.25, 0, 0, 0.25]))
        elif dt < pd.Timestamp("2020-06-01"):
            repo = max(4.0, repo - 0.40)
        elif dt < pd.Timestamp("2022-04-01"):
            repo = 4.0 + np.random.normal(0, 0.05)
        else:
            repo = min(6.75, repo + np.random.choice([0, 0, 0.25, 0.35]))

        # CPI
        if dt > pd.Timestamp("2021-06-01"):
            cpi = min(7.5, cpi + np.random.normal(0.1, 0.3))
        else:
            cpi = max(3.5, cpi + np.random.normal(0, 0.3))

        # Yield curve slope
        slope += np.random.normal(0, 0.15)
        slope  = np.clip(slope, -0.5, 2.5)

        # FII/DII flows
        fii_flow = np.random.normal(-1000 if dt.year == 2022 else 2000, 8000)
        dii_flow = np.random.normal(6000, 3000)

        records.append({
            "date":        dt,
            "repo_rate":   round(repo, 2),
            "cpi":         round(cpi, 2),
            "yield_slope": round(slope, 3),
            "fii_flow_cr": round(fii_flow, 0),
            "dii_flow_cr": round(dii_flow, 0),
        })

    df = pd.DataFrame(records)
    path = f"{RAW_DIR}/macro_data.csv"
    df.to_csv(path, index=False)
    print(f"[macro] {len(df)} months saved → {path}")
    return df


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CommissionLens — Synthetic Data Generator")
    print("=" * 55)

    fund_pairs = generate_fund_pairs(n=250)
    benchmark  = generate_benchmark()
    macro      = generate_macro_data()
    nav        = generate_nav_histories(fund_pairs, benchmark)

    print("\n✅ All raw data generated successfully.")
    print(f"   fund_pairs : {fund_pairs.shape}")
    print(f"   benchmark  : {benchmark.shape}")
    print(f"   macro      : {macro.shape}")
    print(f"   nav        : {nav.shape}")
