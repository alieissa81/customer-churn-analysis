# Customer Churn Analysis

Analysis of 10,000 bank customer records to identify churn drivers by card type and country, predict at-risk customers, and recommend retention strategies.

**Overall churn rate: 20.4%** | Card types: DIAMOND, GOLD, PLATINUM, SILVER | Countries: France, Germany, Spain

---

## Notebooks

Run in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_eda.ipynb](notebooks/01_eda.ipynb) | Data quality, feature distributions, correlation heatmap, churn by categorical features |
| 2 | [02_segment_analysis.ipynb](notebooks/02_segment_analysis.ipynb) | Churn rate by card type & country, cross-tab heatmap, segment summary stats |
| 3 | [03_prediction_model.ipynb](notebooks/03_prediction_model.ipynb) | Logistic Regression + Decision Tree, feature importance, confusion matrix, ROC curves |
| 4 | [04_recommendations.ipynb](notebooks/04_recommendations.ipynb) | Top 5 risk segments, churn drivers, strategic retention playbook |

---

## Project Structure

```
├── notebooks/          # Analysis notebooks (run in order 01→04)
├── skills/
│   ├── utils/          # Shared Python utilities (plotting, scoring, segment helpers)
│   └── superpowers/    # Reference skill files used during project design
├── outputs/
│   ├── figures/        # Generated charts (PNG)
│   └── tables/         # Generated summary tables (CSV)
├── tests/              # pytest unit tests for utility modules
└── docs/               # Design spec and implementation plan
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Key Findings

See [02_segment_analysis.ipynb](notebooks/02_segment_analysis.ipynb) for full segment breakdown and [04_recommendations.ipynb](notebooks/04_recommendations.ipynb) for the retention playbook.
