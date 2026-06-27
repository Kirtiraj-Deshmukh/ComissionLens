"""
fetch_amfi.py
Fetches the complete AMFI mutual fund list and filters for Indian equity funds.
Pairs regular and direct plans by matching fund house + scheme name.
"""

import requests
import pandas as pd
import re
import os

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
OUT_PATH = "/home/claude/commissionlens/data/raw/amfi_funds.csv"

def fetch_amfi_nav_list():
    print("Fetching AMFI fund list...")
    resp = requests.get(AMFI_URL, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")

    records = []
    current_category = ""
    current_amc = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Category header lines (e.g. "Open Ended Schemes(Equity Scheme - ...)")
        if line.startswith("Open Ended") or line.startswith("Close Ended") or line.startswith("Interval"):
            current_category = line
            continue
        # AMC name lines (no semicolons, not a data row)
        if ";" not in line:
            current_amc = line
            continue
        # Data rows: scheme_code;ISIN1;ISIN2;scheme_name;NAV_date;NAV
        parts = line.split(";")
        if len(parts) < 6:
            continue
        try:
            records.append({
                "scheme_code": parts[0].strip(),
                "isin_growth": parts[1].strip(),
                "isin_idcw": parts[2].strip(),
                "scheme_name": parts[3].strip(),
                "nav_date": parts[4].strip(),
                "nav": float(parts[5].strip()) if parts[5].strip() else None,
                "amc": current_amc,
                "category": current_category,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"Total funds fetched: {len(df)}")
    return df


def filter_equity_funds(df):
    """Keep only open-ended equity funds."""
    equity_mask = df["category"].str.contains("Equity", case=False, na=False)
    open_mask   = df["category"].str.contains("Open Ended", case=False, na=False)
    df_eq = df[equity_mask & open_mask].copy()
    print(f"Equity (open-ended) funds: {len(df_eq)}")
    return df_eq


def tag_plan_type(df):
    """Tag each scheme as Direct or Regular based on name."""
    name_upper = df["scheme_name"].str.upper()
    df = df.copy()
    df["plan_type"] = "Unknown"
    df.loc[name_upper.str.contains("DIRECT"), "plan_type"] = "Direct"
    df.loc[name_upper.str.contains("REGULAR"), "plan_type"] = "Regular"
    return df


def pair_direct_regular(df):
    """
    For each Regular plan, try to find its Direct counterpart.
    Returns a DataFrame of matched pairs with both scheme codes.
    """
    direct  = df[df["plan_type"] == "Direct"].copy()
    regular = df[df["plan_type"] == "Regular"].copy()

    def normalise(name):
        """Strip plan/option keywords for matching."""
        n = name.upper()
        for kw in ["DIRECT", "REGULAR", "GROWTH", "IDCW", "DIVIDEND",
                   "PLAN", "OPTION", "-", "–"]:
            n = n.replace(kw, " ")
        return re.sub(r"\s+", " ", n).strip()

    direct["norm"]  = direct["scheme_name"].apply(normalise)
    regular["norm"] = regular["scheme_name"].apply(normalise)

    # Build lookup: norm_name → (scheme_code, scheme_name) for direct funds
    direct_lookup = {}
    for _, row in direct.iterrows():
        direct_lookup.setdefault(row["norm"], []).append(row)

    pairs = []
    for _, reg_row in regular.iterrows():
        matches = direct_lookup.get(reg_row["norm"], [])
        if len(matches) == 1:
            dir_row = matches[0]
            pairs.append({
                "amc": reg_row["amc"],
                "category": reg_row["category"],
                "base_name": reg_row["norm"],
                "regular_code": reg_row["scheme_code"],
                "regular_name": reg_row["scheme_name"],
                "direct_code":  dir_row["scheme_code"],
                "direct_name":  dir_row["scheme_name"],
                "nav_date": reg_row["nav_date"],
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"Matched Regular-Direct pairs: {len(pairs_df)}")
    return pairs_df


def main():
    df_all  = fetch_amfi_nav_list()
    df_eq   = filter_equity_funds(df_all)
    df_eq   = tag_plan_type(df_eq)
    pairs   = pair_direct_regular(df_eq)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pairs.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    print(pairs.head(5).to_string())
    return pairs


if __name__ == "__main__":
    main()
