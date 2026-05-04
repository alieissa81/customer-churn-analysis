# Customer Churn Analysis — Design Spec
**Date:** 2026-05-04  
**Status:** Approved

---

## Overview

A multi-notebook Jupyter project that analyzes 10,000 customer records from `Customer-Churn-Records.csv`, delivers segmented churn analysis by card type and country, builds a simple interpretable churn prediction model, and produces a strategic retention playbook. The project is structured to be published directly to GitHub.

---

## Data

**Source:** `Customer-Churn-Records.csv` (10,000 rows, no PII, synthetic data)

**Key columns:**
- `Geography` — France, Germany, Spain
- `Card Type` — DIAMOND, GOLD, PLATINUM, SILVER
- `Exited` — target variable (1 = churned, 0 = retained)
- `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Complain`, `Satisfaction Score`, `Point Earned`

**Overall churn rate:** 20.4% (2,038 / 10,000)

---

## Folder Structure

```
Customer Churn/
├── README.md
├── requirements.txt
├── Customer-Churn-Records.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_segment_analysis.ipynb
│   ├── 03_prediction_model.ipynb
│   └── 04_recommendations.ipynb
├── skills/
│   ├── utils/
│   │   ├── plotting.py
│   │   ├── churn_scoring.py
│   │   └── segment_helpers.py
│   └── superpowers/
├── outputs/
│   ├── figures/
│   └── tables/
└── docs/
    └── superpowers/
        └── specs/
```

---

## Notebooks

### 01_eda.ipynb — Exploratory Data Analysis
- Data quality check: nulls, dtypes, value counts
- Overall churn rate summary
- Distribution plots: Age, Balance, CreditScore, Tenure, Points Earned
- Correlation heatmap across numeric features
- Churn rate breakdowns: Gender, IsActiveMember, NumOfProducts, HasCrCard

### 02_segment_analysis.ipynb — Analysis by Card Type & Country
- Churn rate per card type (bar chart)
- Churn rate per country (bar chart)
- Cross-tab heatmap: card type × country churn rates
- Per-segment key metrics: avg satisfaction score, avg balance, avg credit score, complaint rate

### 03_prediction_model.ipynb — Churn Prediction Model
- Feature engineering: one-hot encoding for Geography, Gender, Card Type
- Train/test split: 80/20
- Model 1 — Logistic Regression: coefficients table (feature → churn direction + magnitude)
- Model 2 — Decision Tree: visual tree diagram (max_depth=4 for readability)
- Evaluation: accuracy, precision, recall, confusion matrix, ROC-AUC curve
- Feature importance ranking (shared between both models)

### 04_recommendations.ipynb — Strategic Retention Playbook
- Identify top 3–5 highest-risk segments by card type × country combination
- For each segment: churn rate, key churn drivers (from model), 2–3 concrete retention actions
- Summary table: Segment → Risk Level → Root Cause → Recommended Action

---

## Python Utility Modules (`skills/utils/`)

| File | Purpose |
|------|---------|
| `plotting.py` | Standardized chart style: consistent color palette per card type/country, `savefig` helper to export to `outputs/figures/` |
| `churn_scoring.py` | Takes trained model + DataFrame, returns churn probability scores + top contributing features per row |
| `segment_helpers.py` | Filter-by-segment, compute churn rate, summary stats — reused across notebooks 02 and 04 |

---

## Technical Stack

| Library | Version (pinned in requirements.txt) | Purpose |
|---------|--------------------------------------|---------|
| pandas | latest stable | Data manipulation |
| numpy | latest stable | Numerical ops |
| matplotlib | latest stable | Visualizations |
| seaborn | latest stable | Statistical plots |
| scikit-learn | latest stable | ML models + metrics |
| jupyter | latest stable | Notebook runtime |

**No external APIs, databases, or secrets.** Pure CSV-in, notebooks-out.

---

## GitHub Readiness

- All visualizations use matplotlib/seaborn (GitHub renders these natively in notebooks)
- `README.md` serves as project landing page with links to each notebook in order
- `requirements.txt` pins all versions for reproducibility
- CSV included in repo (anonymized synthetic data, no PII)

---

## Success Criteria

1. Each notebook runs top-to-bottom without errors
2. Segmented analysis clearly shows churn rate differences across all 4 card types × 3 countries
3. Model achieves interpretable feature importance (coefficients / tree rules readable)
4. Recommendations section maps each high-risk segment to at least 2 concrete retention actions
5. Project clones and runs cleanly from GitHub via `pip install -r requirements.txt`
