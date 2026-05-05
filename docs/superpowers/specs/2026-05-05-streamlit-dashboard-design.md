# Customer Churn — Streamlit Dashboard Design Spec
**Date:** 2026-05-05  
**Status:** Approved

---

## Overview

A multi-page Streamlit app that turns the existing customer churn analysis into an interactive portfolio dashboard. The app reads from pre-computed CSVs and pickled models — no re-training on load. Deployed to Streamlit Cloud for public access.

---

## Folder Structure

```
streamlit_app/
├── app.py                        # Home page
├── pages/
│   ├── 1_Segment_Explorer.py
│   ├── 2_Model_Insights.py
│   └── 3_Retention_Playbook.py
├── models/
│   ├── lr_model.pkl
│   └── dt_model.pkl
└── requirements.txt              # Streamlit-specific deps
```

The app lives inside the existing project repo. Output CSVs are read from `outputs/tables/`. Pickled models are saved once from the trained notebooks into `streamlit_app/models/`.

---

## Data Layer

All pages load from pre-computed files — no pandas/sklearn computation at runtime beyond simple filtering.

| File | Used by |
|------|---------|
| `outputs/tables/segment_summary.csv` | Segment Explorer |
| `outputs/tables/churn_by_card.csv` | Segment Explorer |
| `outputs/tables/churn_by_country.csv` | Segment Explorer |
| `outputs/tables/top_risk_segments.csv` | Retention Playbook |
| `outputs/tables/retention_playbook.csv` | Retention Playbook |
| `outputs/tables/lr_coefficients.csv` | Model Insights |
| `outputs/tables/feature_importance.csv` | Model Insights |
| `streamlit_app/models/lr_model.pkl` | Model Insights (metrics) |
| `streamlit_app/models/dt_model.pkl` | Model Insights (metrics) |

---

## Pages

### Home (`app.py`)

- Title: "Customer Churn Analysis"
- 3 headline stat columns: 10,000 customers · 20.4% churn rate · 2 prediction models
- 3 navigation cards linking to each page: Segment Explorer, Model Insights, Retention Playbook

### Segment Explorer (`pages/1_Segment_Explorer.py`)

**Sidebar filters:**
- Card Type multiselect (default: All) — DIAMOND, GOLD, PLATINUM, SILVER
- Country multiselect (default: All) — France, Germany, Spain

**Main area (top to bottom):**
1. Two bar charts side by side: Churn Rate by Card Type + Churn Rate by Country — both react to filters
2. Heatmap table: Card Type × Country churn rates, color-coded by risk level
3. Segment summary table: Card Type, Country, Customers, Churn Rate, Avg Balance, Complaint Rate — sortable via `st.dataframe`

### Model Insights (`pages/2_Model_Insights.py`)

**Top row — 4 metric cards:**
- LR Accuracy: 86.3% | LR ROC-AUC: 0.869
- DT Accuracy: 82.6% | DT ROC-AUC: 0.830

**Main area (top to bottom):**
1. Feature importance side by side: LR coefficients bar chart (left) + DT feature importances bar chart (right)
2. ROC curves: both models on one chart with AUC in legend
3. Confusion matrices side by side: LR (left) + DT (right)

No filters. Static display of trained model results.

### Retention Playbook (`pages/3_Retention_Playbook.py`)

1. Top 5 risk segments horizontal bar chart (ranked by churn rate)
2. 5 strategy cards — one per theme:
   - High Complaint Rate → All segments esp. Germany
   - Inactive Members → >60 days inactive
   - 3+ Products → Multi-product holders
   - Germany GOLD & SILVER → DE GOLD, DE SILVER
   - Low Satisfaction → Satisfaction Score < 3

Each card shows: theme title, key driver, target segment, 2–3 recommended actions.

---

## Technical Stack

| Library | Purpose |
|---------|---------|
| streamlit | App framework |
| pandas | Data loading + filtering |
| plotly | Interactive charts (replaces matplotlib for Streamlit) |
| scikit-learn | Pickle model loading |
| pickle | Model deserialisation |

Use Plotly Express for all charts (not matplotlib) — Streamlit renders Plotly charts interactively without extra config.

---

## Deployment

- **Platform:** Streamlit Cloud (streamlit.io/cloud)
- **Entry point:** `streamlit_app/app.py`
- **Dependencies:** `streamlit_app/requirements.txt`
- **Data files:** committed to repo (outputs/tables/ CSVs + streamlit_app/models/ pickles)
- No secrets or environment variables needed.

---

## Success Criteria

1. `streamlit run streamlit_app/app.py` starts without errors
2. Segment Explorer filters update all 3 components live
3. Model Insights displays correct metric values matching notebook outputs
4. Retention Playbook renders all 5 strategy cards
5. App deploys and runs on Streamlit Cloud from the GitHub repo
