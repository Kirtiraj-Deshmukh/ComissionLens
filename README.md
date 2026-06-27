# CommissionLens 

**Commission-Adjusted Alpha Prediction in Indian Mutual Funds**

> *Is your regular mutual fund plan actually worth its commission cost?*

---

## Problem Statement

India has 90M+ demat accounts, yet most retail investors in regular plans pay 0.5–1.5% annually as distributor commission over their direct plan equivalents. On a ₹5,000/month SIP over 20 years, this erodes ₹8–12 lakh. This project builds an ML framework to predict whether a fund will generate sufficient net alpha to justify that commission.

---

## Pipeline Architecture

```
Raw Data                Feature Engineering          ML Models
─────────               ───────────────────          ─────────
AMFI NAV API    ──►    Quarterly alpha              XGBoost Regressor
mfapi.in        ──►    Rolling Sharpe (4Q, 8Q)  ──► (predict net alpha)
RBI DBIE        ──►    Beta, Info Ratio
NSE Nifty 50    ──►    Alpha persistence        ──► XGBoost Classifier
                        Macro regime vars             (justify? 0/1)
                                                        │
                                                        ▼
                                                    SHAP Analysis
                                                    SIP Backtest
                                                    XIRR Comparison
```

---

## Results

| Strategy | XIRR (2019–2023) | Final Value (₹5K/mo) |
|---|---|---|
| Model-Guided Regular | **20.22%** | ₹4.29L |
| Direct Benchmark | 19.91% | ₹4.85L |
| Naive Regular | 18.70% | ₹4.71L |

- Walk-Forward AUC: **~0.53** (mean across 20 quarters)
- Top predictors: RBI Repo Rate, Beta, Alpha Persistence, Rolling Sharpe

---

## Project Structure

```
commissionlens/
├── data/
│   ├── raw/                    # AMFI, NAV, macro, benchmark CSVs/parquet
│   └── processed/              # Feature matrix, model predictions
├── notebooks/
│   └── CommissionLens.ipynb    # Full pipeline notebook (end-to-end)
├── src/
│   ├── generate_synthetic_data.py   # Realistic synthetic data (sandbox)
│   ├── fetch_real_data.py           # Real API fetchers (run locally)
│   ├── feature_engineering.py       # Quarterly feature matrix builder
│   ├── ml_model.py                  # XGBoost training (final train/test)
│   ├── walk_forward_cv.py           # Expanding-window OOS predictions
│   ├── shap_analysis.py             # SHAP explainability + charts
│   └── sip_backtest.py              # XIRR SIP comparison
└── reports/
    ├── shap_summary_regression.png
    ├── shap_summary_classification.png
    ├── shap_beeswarm_classification.png
    ├── sip_backtest_chart.png
    ├── xirr_comparison.png
    ├── feature_importance.csv
    ├── metrics.json
    ├── shap_report.txt
    └── sip_backtest_summary.txt
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install xgboost shap pandas numpy scipy yfinance pyarrow matplotlib seaborn tqdm

# 2. Generate synthetic data (or fetch real data — see src/fetch_real_data.py)
python src/generate_synthetic_data.py

# 3. Feature engineering
python src/feature_engineering.py

# 4. Walk-forward CV (generates OOS predictions for full backtest period)
python src/walk_forward_cv.py

# 5. Final model + SHAP
python src/ml_model.py
python src/shap_analysis.py

# 6. SIP backtest
python src/sip_backtest.py

# 7. Open the notebook for the full story
jupyter notebook notebooks/CommissionLens.ipynb
```

---

## Using Real Data

Replace the synthetic data step with:
```bash
python src/fetch_real_data.py  # see instructions inside
```

Manual downloads needed:
- **RBI DBIE**: [dbie.rbi.org.in](https://dbie.rbi.org.in) → Repo rate, CPI
- **NSE FII/DII**: [nseindia.com](https://www.nseindia.com/reports-indices-equity-fii-dii-data)

mfapi.in (NAV history) and AMFI (fund list) are fetched automatically.

---

## Key Concepts

| Term | Definition |
|---|---|
| **Net Alpha** | Gross Return − Benchmark Return − Expense Gap |
| **Commission Justified** | Net Alpha > 0 |
| **XIRR** | Extended IRR for irregular SIP cashflows |
| **Walk-Forward CV** | Expanding-window train/test to avoid look-ahead bias |
| **Information Ratio** | Alpha / Tracking Error (annualised) |
| **Alpha Persistence** | AR(1) of quarterly alpha (does past alpha predict future?) |

---

*ML · FinTech · Python · Indian Markets*
