"""
fetch_real_data.py
------------------
Real data fetchers for CommissionLens.
Run this script LOCALLY (where mfapi.in and amfiindia.com are accessible).

APIs used:
  1. AMFI NAVAll.txt       → fund list, expense ratios (sort of)
  2. mfapi.in              → historical NAV per fund (free, no key needed)
  3. NSF via yfinance      → Nifty 50 daily closes (^NSEI)
  4. RBI DBIE              → repo rate, CPI (requires scraping or manual download)

Usage:
  pip install requests pandas yfinance tqdm pyarrow
  python fetch_real_data.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. AMFI Fund List
# ─────────────────────────────────────────────

def fetch_amfi_fund_list():
    """
    Fetches AMFI NAVAll.txt and returns equity fund pairs.
    NOTE: AMFI doesn't directly publish expense ratios in NAVAll.txt.
    For expense ratios, scrape individual fund pages or use a third-party source.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    # ... (same parsing as in fetch_amfi.py)
    print("AMFI fund list fetched.")


# ─────────────────────────────────────────────
# 2. mfapi.in — Historical NAVs
# ─────────────────────────────────────────────

def fetch_nav_mfapi(scheme_code: int, retries: int = 3) -> pd.DataFrame:
    """
    Fetches historical NAV for a given scheme code from mfapi.in.
    Returns DataFrame with columns: date, nav
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            records = [
                {"date": pd.to_datetime(r["date"], format="%d-%m-%Y"), "nav": float(r["nav"])}
                for r in data.get("data", [])
                if r.get("nav") and r["nav"] != "N.A."
            ]
            df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Failed scheme {scheme_code}: {e}")
                return pd.DataFrame()
            time.sleep(1)


def fetch_all_navs(fund_pairs: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    """
    Fetches NAV histories for all fund pairs (both regular and direct).
    Rate limited to ~2 requests/sec to be polite to mfapi.in.
    """
    all_records = []
    for _, row in tqdm(fund_pairs.iterrows(), total=len(fund_pairs), desc="Fetching NAVs"):
        for plan_col, code_col in [("regular", "regular_code"), ("direct", "direct_code")]:
            code = int(row[code_col])
            nav_df = fetch_nav_mfapi(code)
            if nav_df.empty:
                continue
            nav_df["fund_id"]   = row["fund_id"]
            nav_df["plan_type"] = plan_col
            all_records.append(nav_df)
            time.sleep(delay)

    if not all_records:
        print("No NAV data fetched!")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)

    # Pivot to wide format: date, fund_id, nav_regular, nav_direct
    wide = combined.pivot_table(
        index=["date", "fund_id"],
        columns="plan_type",
        values="nav",
        aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"regular": "nav_regular", "direct": "nav_direct"})

    path = f"{RAW_DIR}/nav_histories.parquet"
    wide.to_parquet(path, index=False)
    print(f"NAV histories saved → {path} ({wide.shape})")
    return wide


# ─────────────────────────────────────────────
# 3. Nifty 50 via yfinance
# ─────────────────────────────────────────────

def fetch_nifty50(start="2018-01-01", end="2024-06-30") -> pd.DataFrame:
    """
    Downloads Nifty 50 daily closes from Yahoo Finance.
    Ticker: ^NSEI
    """
    import yfinance as yf
    nifty = yf.download("^NSEI", start=start, end=end, progress=False)
    nifty = nifty[["Close"]].reset_index()
    nifty.columns = ["date", "nifty50_close"]
    nifty["nifty50_return"] = nifty["nifty50_close"].pct_change()
    path = f"{RAW_DIR}/benchmark_nifty50.csv"
    nifty.to_csv(path, index=False)
    print(f"Nifty 50 saved → {path} ({nifty.shape})")
    return nifty


# ─────────────────────────────────────────────
# 4. RBI DBIE Macro Data
# ─────────────────────────────────────────────

def fetch_rbi_macro() -> pd.DataFrame:
    """
    RBI DBIE (dbie.rbi.org.in) does not have a clean public JSON API.
    Options:
      A) Manual download: go to https://dbie.rbi.org.in/DBIE/dbie.rbi
         → Monetary → Repo Rate → Download CSV
         → Prices → CPI → Download CSV
      B) Use the 'dbieapi' Python package if available
      C) Use RBI press releases (structured) scraped via BeautifulSoup

    Below is a template assuming you've manually downloaded the CSVs.
    Expected columns: date, repo_rate, cpi
    """
    # Placeholder — replace with actual RBI CSV path
    rbi_path = "data/raw/rbi_macro_manual.csv"
    if os.path.exists(rbi_path):
        df = pd.read_csv(rbi_path, parse_dates=["date"])
        print(f"RBI macro loaded: {df.shape}")
        return df
    else:
        print(f"⚠ RBI CSV not found at {rbi_path}.")
        print("  Download from https://dbie.rbi.org.in/DBIE/dbie.rbi and save as CSV.")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 5. FII / DII Flows (SEBI/NSE)
# ─────────────────────────────────────────────

def fetch_fii_dii_flows() -> pd.DataFrame:
    """
    FII/DII data:
      NSE India: https://www.nseindia.com/reports-indices-equity-fii-dii-data
      Alternatively: SEBI monthly bulletin (scraped)
    
    Below is a template. NSE data is available as CSV download from their site.
    """
    nse_path = "data/raw/nse_fii_dii.csv"
    if os.path.exists(nse_path):
        df = pd.read_csv(nse_path, parse_dates=["date"])
        print(f"FII/DII data loaded: {df.shape}")
        return df
    else:
        print(f"⚠ NSE FII/DII CSV not found at {nse_path}.")
        print("  Download from https://www.nseindia.com/reports-indices-equity-fii-dii-data")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("CommissionLens — Real Data Fetcher")
    print("=" * 45)
    print("Steps to run:")
    print("1. Run fetch_amfi_fund_list() to get equity fund pairs")
    print("2. Run fetch_all_navs(fund_pairs) — takes ~30 min for 250 funds")
    print("3. Run fetch_nifty50() for benchmark data")
    print("4. Manually download RBI DBIE CSV and NSE FII/DII CSV")
    print("5. Then run feature_engineering.py → walk_forward_cv.py → sip_backtest.py")
    print()
    print("NOTE: mfapi.in has 250 funds × 2 plans = 500 API calls.")
    print("      With 0.5s delay, this takes ~4 minutes. Be respectful of the free API.")
