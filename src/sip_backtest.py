"""
sip_backtest.py
---------------
SIP back-validation: 2019 Q1 → 2023 Q4

Strategy A (Model-guided): Each quarter, invest ₹5,000 SIP into the
  top-decile funds ranked by predicted commission justification probability.
  Use REGULAR plan NAV (since that's what retail SIP investors buy).

Strategy B (Naive):  Invest ₹5,000 SIP split equally across ALL 250
  regular plans — simulates unguided regular plan investing.

Strategy C (Direct): Invest ₹5,000 SIP split equally across all 250
  DIRECT plans — the counterfactual (zero commission cost benchmark).

Outputs:
  - reports/sip_backtest_results.csv
  - reports/sip_backtest_chart.png
  - reports/sip_backtest_summary.txt
  - reports/xirr_comparison.png
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

PROC_DIR   = "/home/claude/commissionlens/data/processed"
RAW_DIR    = "/home/claude/commissionlens/data/raw"
REPORT_DIR = "/home/claude/commissionlens/reports"

STYLE = {
    "bg":    "#0a0a0a",
    "panel": "#111111",
    "gold":  "#d4a017",
    "green": "#39ff14",
    "red":   "#ff4444",
    "blue":  "#4fc3f7",
    "text":  "#e8e8e8",
    "grid":  "#1f1f1f",
}

MONTHLY_SIP = 5000
BACKTEST_START = pd.Timestamp("2019-01-01")
BACKTEST_END   = pd.Timestamp("2023-12-31")


# ─────────────────────────────────────────────
# XIRR
# ─────────────────────────────────────────────

def xirr(cashflows: list[tuple]) -> float:
    """
    cashflows: list of (date, amount) where investments are negative,
               final portfolio value is positive.
    Returns annualised IRR.
    """
    if not cashflows:
        return 0.0

    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = min(dates)
    days    = np.array([(d - t0).days for d in dates], dtype=float)

    def npv(r):
        return sum(a / (1 + r) ** (d / 365.25)
                   for a, d in zip(amounts, days))

    try:
        return brentq(npv, -0.9, 10.0, maxiter=1000)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────

def load_data():
    nav_df   = pd.read_parquet(f"{RAW_DIR}/nav_histories.parquet")
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    preds_df = pd.read_parquet(f"{PROC_DIR}/test_predictions.parquet")
    preds_df["quarter"] = pd.to_datetime(preds_df["quarter"])
    fund_df  = pd.read_csv(f"{RAW_DIR}/fund_pairs.csv")
    return nav_df, preds_df, fund_df


# ─────────────────────────────────────────────
# SIP Simulation
# ─────────────────────────────────────────────

def simulate_sip(nav_df, fund_ids, nav_col, sip_per_fund_per_month,
                 start=BACKTEST_START, end=BACKTEST_END):
    """
    Monthly SIP into a basket of fund_ids using nav_col (nav_regular / nav_direct).
    Returns:
      cashflows: list of (date, amount)  — investments negative, final value positive
      monthly_portfolio_value: pd.Series
    """
    # Monthly last-day NAVs for selected funds
    nav_basket = (
        nav_df[nav_df["fund_id"].isin(fund_ids) &
               (nav_df["date"] >= start) &
               (nav_df["date"] <= end)]
        .set_index(["date", "fund_id"])[nav_col]
        .unstack("fund_id")
        .resample("ME").last()
        .ffill()
    )

    months     = nav_basket.index
    units_held = pd.Series(0.0, index=fund_ids)
    cashflows  = []
    portfolio_vals = []

    for month in months:
        nav_today = nav_basket.loc[month]

        # Buy SIP
        for fid in fund_ids:
            if fid in nav_today.index and not np.isnan(nav_today[fid]) and nav_today[fid] > 0:
                units_bought = sip_per_fund_per_month / nav_today[fid]
                units_held[fid] += units_bought

        cashflows.append((month, -sip_per_fund_per_month * len(fund_ids)))

        # Portfolio value
        pv = sum(
            units_held[fid] * nav_today[fid]
            for fid in fund_ids
            if fid in nav_today.index and not np.isnan(nav_today[fid])
        )
        portfolio_vals.append(pv)

    # Final redemption
    last_nav   = nav_basket.iloc[-1]
    final_value = sum(
        units_held[fid] * last_nav[fid]
        for fid in fund_ids
        if fid in last_nav.index and not np.isnan(last_nav[fid])
    )
    cashflows.append((months[-1], final_value))

    return cashflows, pd.Series(portfolio_vals, index=months)


# ─────────────────────────────────────────────
# Strategy A: Model-Guided (top decile each quarter)
# ─────────────────────────────────────────────

def model_guided_basket(preds_df, fund_df):
    """
    Each quarter, pick top-10% funds by predicted commission justification probability.
    Returns: dict of month → list of fund_ids to invest in that month.
    """
    quarters = preds_df["quarter"].unique()
    basket_by_quarter = {}
    for q in sorted(quarters):
        qdf = preds_df[preds_df["quarter"] == q].copy()
        n_top = max(1, len(qdf) // 10)
        top_funds = (
            qdf.nlargest(n_top, "pred_justified_proba")["fund_id"].tolist()
        )
        basket_by_quarter[q] = top_funds
    return basket_by_quarter


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("CommissionLens — SIP Back-Validation")
    print("=" * 55)

    nav_df, preds_df, fund_df = load_data()
    all_fund_ids = fund_df["fund_id"].tolist()

    # Restrict to backtest window
    preds_bt = preds_df[
        (preds_df["quarter"] >= BACKTEST_START) &
        (preds_df["quarter"] <= BACKTEST_END)
    ]

    # ── Strategy B: Naive Regular ──
    print("\nRunning Strategy B — Naive Regular (all funds)...")
    sip_per_fund_b = MONTHLY_SIP / len(all_fund_ids)
    cf_b, pv_b = simulate_sip(nav_df, all_fund_ids, "nav_regular",
                               sip_per_fund_b)
    xirr_b = xirr(cf_b)
    print(f"  XIRR (Naive Regular): {xirr_b:.2%}")

    # ── Strategy C: Direct Benchmark ──
    print("Running Strategy C — Direct Benchmark (all funds)...")
    sip_per_fund_c = MONTHLY_SIP / len(all_fund_ids)
    cf_c, pv_c = simulate_sip(nav_df, all_fund_ids, "nav_direct",
                               sip_per_fund_c)
    xirr_c = xirr(cf_c)
    print(f"  XIRR (Direct):        {xirr_c:.2%}")

    # ── Strategy A: Model-Guided ──
    print("Running Strategy A — Model-Guided Regular (top-decile)...")
    basket_map = model_guided_basket(preds_bt, fund_df)

    # For simplicity: use last predicted quarter's basket throughout (or rotate)
    # We build monthly cashflows using rotating quarterly baskets
    months_bt = pd.date_range(BACKTEST_START, BACKTEST_END, freq="ME")
    quarters_sorted = sorted(basket_map.keys())

    def get_basket_for_month(month):
        for q in reversed(quarters_sorted):
            if q <= month:
                return basket_map[q]
        return basket_map[quarters_sorted[0]]

    # Combine by month
    units_a  = {}
    cf_a     = []
    pv_a_list = []

    # Build all unique fund ids needed
    all_basket_ids = list(set(fid for fids in basket_map.values() for fid in fids))
    nav_lookup = (
        nav_df[nav_df["fund_id"].isin(all_basket_ids) &
               (nav_df["date"] >= BACKTEST_START) &
               (nav_df["date"] <= BACKTEST_END)]
        .set_index(["date", "fund_id"])["nav_regular"]
        .unstack("fund_id")
        .resample("ME").last()
        .ffill()
    )

    for month in months_bt:
        if month not in nav_lookup.index:
            continue
        basket = get_basket_for_month(month)
        nav_today = nav_lookup.loc[month]
        valid_basket = [f for f in basket if f in nav_today.index and not np.isnan(nav_today[f])]
        if not valid_basket:
            continue

        sip_per = MONTHLY_SIP / len(valid_basket)
        for fid in valid_basket:
            if fid not in units_a:
                units_a[fid] = 0.0
            units_a[fid] += sip_per / nav_today[fid]

        cf_a.append((month, -MONTHLY_SIP))

        pv = sum(units_a.get(fid, 0) * nav_today[fid] for fid in valid_basket)
        pv_a_list.append(pv)

    # Final value (use last basket)
    last_month = months_bt[-1]
    if last_month in nav_lookup.index:
        last_basket = get_basket_for_month(last_month)
        nav_last = nav_lookup.loc[last_month]
        final_val_a = sum(
            units_a.get(fid, 0) * nav_last[fid]
            for fid in last_basket
            if fid in nav_last.index and not np.isnan(nav_last[fid])
        )
        cf_a.append((last_month, final_val_a))

    xirr_a = xirr(cf_a)
    pv_a = pd.Series(pv_a_list, index=months_bt[:len(pv_a_list)])
    print(f"  XIRR (Model-Guided):  {xirr_a:.2%}")

    # ─────────────────────────────────────────────
    # Invested amount series
    # ─────────────────────────────────────────────
    n_months    = len(months_bt)
    invested    = pd.Series(
        [MONTHLY_SIP * (i + 1) for i in range(n_months)],
        index=months_bt
    )

    # ─────────────────────────────────────────────
    # Portfolio Value Chart
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])
    ax.spines[:].set_color(STYLE["gold"])
    ax.tick_params(colors=STYLE["text"])

    ax.plot(pv_a.index, pv_a.values / 1e5,
            color=STYLE["gold"], linewidth=2.5, label=f"Model-Guided (XIRR {xirr_a:.1%})")
    ax.plot(pv_b.index, pv_b.values / 1e5,
            color=STYLE["red"], linewidth=1.8, linestyle="--",
            label=f"Naive Regular (XIRR {xirr_b:.1%})")
    ax.plot(pv_c.index, pv_c.values / 1e5,
            color=STYLE["green"], linewidth=1.8, linestyle=":",
            label=f"Direct Benchmark (XIRR {xirr_c:.1%})")
    ax.plot(invested.index, invested.values / 1e5,
            color="#666666", linewidth=1, linestyle="-.",
            label="Amount Invested")

    ax.set_title("CommissionLens — SIP Back-Validation 2019–2023\n"
                 "₹5,000/month SIP | Portfolio Value (₹ Lakh)",
                 color=STYLE["gold"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", color=STYLE["text"])
    ax.set_ylabel("Portfolio Value (₹ Lakh)", color=STYLE["text"])
    ax.legend(facecolor=STYLE["panel"], edgecolor=STYLE["gold"],
              labelcolor=STYLE["text"], fontsize=10)
    ax.grid(color=STYLE["grid"], linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))
    plt.tight_layout()
    fig.savefig(f"{REPORT_DIR}/sip_backtest_chart.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"\n  Saved sip_backtest_chart.png")

    # ─────────────────────────────────────────────
    # XIRR Comparison Bar Chart
    # ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=STYLE["bg"])
    ax2.set_facecolor(STYLE["panel"])
    ax2.spines[:].set_color(STYLE["gold"])
    ax2.tick_params(colors=STYLE["text"])

    strategies = ["Naive Regular", "Model-Guided\n(Regular)", "Direct\nBenchmark"]
    xirrs      = [xirr_b, xirr_a, xirr_c]
    colors     = [STYLE["red"], STYLE["gold"], STYLE["green"]]
    bars       = ax2.bar(strategies, [x * 100 for x in xirrs],
                         color=colors, width=0.5, edgecolor=STYLE["bg"], linewidth=1.5)

    for bar, val in zip(bars, xirrs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val:.2%}", ha="center", va="bottom",
                 color=STYLE["text"], fontsize=12, fontweight="bold")

    ax2.set_title("XIRR Comparison — 2019-2023 SIP Backtest",
                  color=STYLE["gold"], fontsize=13, fontweight="bold")
    ax2.set_ylabel("XIRR (%)", color=STYLE["text"])
    ax2.grid(axis="y", color=STYLE["grid"], linewidth=0.5)
    plt.tight_layout()
    fig2.savefig(f"{REPORT_DIR}/xirr_comparison.png", dpi=150,
                 bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"  Saved xirr_comparison.png")

    # ─────────────────────────────────────────────
    # Text Summary
    # ─────────────────────────────────────────────
    total_invested  = MONTHLY_SIP * n_months
    final_a = pv_a.iloc[-1] if len(pv_a) > 0 else 0
    final_b = pv_b.iloc[-1] if len(pv_b) > 0 else 0
    final_c = pv_c.iloc[-1] if len(pv_c) > 0 else 0

    summary = f"""
CommissionLens — SIP Back-Validation Summary (2019–2023)
=========================================================
Monthly SIP : ₹{MONTHLY_SIP:,}
Period      : {BACKTEST_START.date()} → {BACKTEST_END.date()}
Months      : {n_months}
Total Invested : ₹{total_invested:,.0f}

STRATEGY A — Model-Guided (Top-Decile Regular Plans)
  Final Value : ₹{final_a:,.0f}
  Gain        : ₹{final_a - total_invested:,.0f} ({(final_a/total_invested - 1)*100:.1f}%)
  XIRR        : {xirr_a:.2%}

STRATEGY B — Naive (All Regular Plans)
  Final Value : ₹{final_b:,.0f}
  Gain        : ₹{final_b - total_invested:,.0f} ({(final_b/total_invested - 1)*100:.1f}%)
  XIRR        : {xirr_b:.2%}

STRATEGY C — Direct Plans (No Commission, Counterfactual)
  Final Value : ₹{final_c:,.0f}
  Gain        : ₹{final_c - total_invested:,.0f} ({(final_c/total_invested - 1)*100:.1f}%)
  XIRR        : {xirr_c:.2%}

VERDICT
  Model-guided vs Naive:  Δ XIRR = {(xirr_a - xirr_b)*100:.2f}%
  Direct vs Naive Regular: Δ XIRR = {(xirr_c - xirr_b)*100:.2f}%
  (Positive Δ = model adds value over naive regular investing)
"""
    print(summary)
    with open(f"{REPORT_DIR}/sip_backtest_summary.txt", "w") as f:
        f.write(summary)
    print(f"\n✅ SIP backtest complete. Results saved to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
