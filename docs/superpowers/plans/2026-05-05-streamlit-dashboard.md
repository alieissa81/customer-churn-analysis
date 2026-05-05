# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-page Streamlit portfolio dashboard that makes the existing customer churn analysis interactive and deploys to Streamlit Cloud.

**Architecture:** Multi-page Streamlit app (`streamlit_app/`) reading from pre-computed CSVs in `outputs/tables/` and static PNGs in `outputs/figures/`. A shared `data_loader.py` module handles all CSV reads with `@st.cache_data`. No model re-training on load — LR/DT metrics are hard-coded constants; feature importance charts are built from existing CSVs; ROC curves and confusion matrices are shown as static PNGs.

**Tech Stack:** Streamlit ≥ 1.32, Plotly Express (interactive bar charts + heatmap), pandas (filtering), pytest + streamlit.testing.v1 (tests)

---

## File Map

| File | Responsibility |
|------|---------------|
| `streamlit_app/requirements.txt` | Pinned Streamlit-specific deps for Streamlit Cloud |
| `streamlit_app/conftest.py` | Adds `streamlit_app/` to `sys.path` for all tests |
| `streamlit_app/data_loader.py` | `@st.cache_data` CSV loaders + path constants |
| `streamlit_app/app.py` | Home page — headline stats + navigation cards |
| `streamlit_app/pages/1_Segment_Explorer.py` | Interactive filters → bar charts + heatmap + table |
| `streamlit_app/pages/2_Model_Insights.py` | Static metrics + Plotly feature charts + PNG images |
| `streamlit_app/pages/3_Retention_Playbook.py` | Top risk bar chart + 5 strategy cards |
| `streamlit_app/tests/test_data_loader.py` | Unit tests for all 6 loader functions |
| `streamlit_app/tests/test_pages.py` | AppTest smoke tests — each page loads without exception |

**Existing files read (not modified):**
- `outputs/tables/card_type_summary.csv` — columns: `Card Type`, `churn_rate`, `n_customers`, `avg_balance`, `complaint_rate`
- `outputs/tables/country_summary.csv` — columns: `Geography`, `churn_rate`, `n_customers`, `avg_balance`, `complaint_rate`
- `outputs/tables/cross_segment_summary.csv` — columns: `Card Type`, `Geography`, `churn_rate`, `n_customers`, `avg_balance`, `complaint_rate`
- `outputs/tables/lr_coefficients.csv` — columns: `feature`, `coefficient`
- `outputs/tables/feature_importance.csv` — columns: `feature`, `importance`
- `outputs/tables/retention_playbook.csv` — columns: `Theme`, `Key Driver`, `Target Segments`, `Recommended Action`
- `outputs/figures/03_roc_curves.png`
- `outputs/figures/03_confusion_matrices.png`

---

## Task 1: Requirements + Install

**Files:**
- Create: `streamlit_app/requirements.txt`

- [ ] **Step 1: Create requirements file**

```
streamlit==1.45.0
plotly==6.1.1
pandas==2.3.0
```

- [ ] **Step 2: Install dependencies**

Run: `pip install streamlit==1.45.0 plotly==6.1.1`
Expected: Both install without errors. (`pandas==2.3.0` is already installed.)

- [ ] **Step 3: Verify install**

Run: `python3 -c "import streamlit; import plotly; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add streamlit_app/requirements.txt
git commit -m "chore: add streamlit app requirements"
```

---

## Task 2: Data Loader + conftest + Tests

**Files:**
- Create: `streamlit_app/data_loader.py`
- Create: `streamlit_app/conftest.py`
- Create: `streamlit_app/tests/test_data_loader.py`

- [ ] **Step 1: Write the failing tests**

Create `streamlit_app/tests/test_data_loader.py`:

```python
import pytest
import pandas as pd
from data_loader import (
    load_card_type_summary,
    load_country_summary,
    load_cross_segment_summary,
    load_lr_coefficients,
    load_feature_importance,
    load_retention_playbook,
)


def test_card_type_summary_columns():
    df = load_card_type_summary()
    assert {"Card Type", "churn_rate", "n_customers", "avg_balance", "complaint_rate"}.issubset(df.columns)


def test_card_type_summary_row_count():
    df = load_card_type_summary()
    assert len(df) == 4


def test_country_summary_row_count():
    df = load_country_summary()
    assert len(df) == 3


def test_cross_segment_summary_row_count():
    df = load_cross_segment_summary()
    assert len(df) == 12


def test_churn_rates_in_range():
    for loader in [load_card_type_summary, load_country_summary, load_cross_segment_summary]:
        df = loader()
        assert df["churn_rate"].between(0, 1).all(), f"churn_rate out of range in {loader.__name__}"


def test_lr_coefficients_columns():
    df = load_lr_coefficients()
    assert {"feature", "coefficient"}.issubset(df.columns)


def test_feature_importance_columns():
    df = load_feature_importance()
    assert {"feature", "importance"}.issubset(df.columns)


def test_retention_playbook_columns():
    df = load_retention_playbook()
    assert {"Theme", "Key Driver", "Target Segments", "Recommended Action"}.issubset(df.columns)


def test_retention_playbook_has_5_themes():
    df = load_retention_playbook()
    assert df["Theme"].nunique() == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest streamlit_app/tests/test_data_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'data_loader'`

- [ ] **Step 3: Create conftest.py**

Create `streamlit_app/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Step 4: Create data_loader.py**

Create `streamlit_app/data_loader.py`:

```python
from pathlib import Path
import pandas as pd
import streamlit as st

_TABLES = Path(__file__).parent.parent / "outputs" / "tables"


@st.cache_data
def load_card_type_summary() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "card_type_summary.csv")


@st.cache_data
def load_country_summary() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "country_summary.csv")


@st.cache_data
def load_cross_segment_summary() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "cross_segment_summary.csv")


@st.cache_data
def load_lr_coefficients() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "lr_coefficients.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "feature_importance.csv")


@st.cache_data
def load_retention_playbook() -> pd.DataFrame:
    return pd.read_csv(_TABLES / "retention_playbook.csv")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest streamlit_app/tests/test_data_loader.py -v`
Expected: All 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/conftest.py streamlit_app/data_loader.py streamlit_app/tests/test_data_loader.py
git commit -m "feat: add streamlit data loader with tests"
```

---

## Task 3: Home Page

**Files:**
- Create: `streamlit_app/app.py`

- [ ] **Step 1: Write the failing smoke test**

Add to `streamlit_app/tests/test_pages.py` (create the file):

```python
from pathlib import Path
from streamlit.testing.v1 import AppTest

_APP_DIR = Path(__file__).parent.parent


def test_home_loads_without_exception():
    at = AppTest.from_file(str(_APP_DIR / "app.py"))
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest streamlit_app/tests/test_pages.py::test_home_loads_without_exception -v`
Expected: `FileNotFoundError` or similar — `app.py` not found

- [ ] **Step 3: Create app.py**

Create `streamlit_app/app.py`:

```python
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Churn Analysis")
st.markdown(
    "Analysis of **10,000 bank customer records** to identify churn drivers "
    "and retention strategies."
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", "10,000")
col2.metric("Overall Churn Rate", "20.4%")
col3.metric("Prediction Models", "2")

st.markdown("---")
st.subheader("Explore the Analysis")

c1, c2, c3 = st.columns(3)
with c1:
    st.info(
        "**📊 Segment Explorer**\n\n"
        "Filter by card type & country to explore churn rates interactively."
    )
with c2:
    st.info(
        "**🤖 Model Insights**\n\n"
        "Logistic Regression + Decision Tree performance, coefficients, and feature importance."
    )
with c3:
    st.info(
        "**📋 Retention Playbook**\n\n"
        "5 strategic themes with targeted actions for the highest-risk segments."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest streamlit_app/tests/test_pages.py::test_home_loads_without_exception -v`
Expected: PASS

- [ ] **Step 5: Verify locally in browser**

Run: `streamlit run streamlit_app/app.py`
Open: http://localhost:8501
Expected: Home page renders with 3 metric cards and 3 info cards. Stop the server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/app.py streamlit_app/tests/test_pages.py
git commit -m "feat: add streamlit home page"
```

---

## Task 4: Segment Explorer

**Files:**
- Create: `streamlit_app/pages/1_Segment_Explorer.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `streamlit_app/tests/test_pages.py`:

```python
def test_segment_explorer_loads_without_exception():
    at = AppTest.from_file(str(_APP_DIR / "pages" / "1_Segment_Explorer.py"))
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest streamlit_app/tests/test_pages.py::test_segment_explorer_loads_without_exception -v`
Expected: `FileNotFoundError` — page not found

- [ ] **Step 3: Create 1_Segment_Explorer.py**

Create `streamlit_app/pages/1_Segment_Explorer.py`:

```python
import streamlit as st
import plotly.express as px
from data_loader import (
    load_card_type_summary,
    load_country_summary,
    load_cross_segment_summary,
)

st.set_page_config(page_title="Segment Explorer", layout="wide")
st.title("📊 Segment Explorer")

CARD_COLORS = {
    "DIAMOND": "#4e79a7",
    "GOLD": "#f28e2b",
    "PLATINUM": "#76b7b2",
    "SILVER": "#e15759",
}
COUNTRY_COLORS = {
    "Germany": "#e15759",
    "Spain": "#76b7b2",
    "France": "#4e79a7",
}

with st.sidebar:
    st.header("Filters")
    all_cards = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
    all_countries = ["France", "Germany", "Spain"]
    selected_cards = st.multiselect("Card Type", all_cards, default=all_cards)
    selected_countries = st.multiselect("Country", all_countries, default=all_countries)

if not selected_cards:
    selected_cards = all_cards
if not selected_countries:
    selected_countries = all_countries

card_df = load_card_type_summary()
country_df = load_country_summary()
cross_df = load_cross_segment_summary()

card_filtered = card_df[card_df["Card Type"].isin(selected_cards)].copy()
country_filtered = country_df[country_df["Geography"].isin(selected_countries)].copy()
cross_filtered = cross_df[
    cross_df["Card Type"].isin(selected_cards) & cross_df["Geography"].isin(selected_countries)
].copy()

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        card_filtered.sort_values("churn_rate"),
        x="churn_rate",
        y="Card Type",
        orientation="h",
        title="Churn Rate by Card Type",
        labels={"churn_rate": "Churn Rate", "Card Type": ""},
        color="Card Type",
        color_discrete_map=CARD_COLORS,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.bar(
        country_filtered.sort_values("churn_rate"),
        x="churn_rate",
        y="Geography",
        orientation="h",
        title="Churn Rate by Country",
        labels={"churn_rate": "Churn Rate", "Geography": ""},
        color="Geography",
        color_discrete_map=COUNTRY_COLORS,
    )
    fig2.update_layout(showlegend=False)
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Churn Rate Heatmap — Card Type × Country")
if not cross_filtered.empty:
    pivot = cross_filtered.pivot(index="Card Type", columns="Geography", values="churn_rate")
    fig3 = px.imshow(
        pivot,
        text_auto=".1%",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Churn Rate by Segment",
    )
    fig3.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data for the selected filters.")

st.subheader("Segment Summary")
if not cross_filtered.empty:
    display = cross_filtered.sort_values("churn_rate", ascending=False)[
        ["Card Type", "Geography", "n_customers", "churn_rate", "avg_balance", "complaint_rate"]
    ].copy()
    display = display.rename(columns={
        "Geography": "Country",
        "n_customers": "Customers",
        "churn_rate": "Churn Rate",
        "avg_balance": "Avg Balance",
        "complaint_rate": "Complaint Rate",
    })
    display["Churn Rate"] = display["Churn Rate"].map("{:.1%}".format)
    display["Avg Balance"] = display["Avg Balance"].map("€{:,.0f}".format)
    display["Complaint Rate"] = display["Complaint Rate"].map("{:.1%}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest streamlit_app/tests/test_pages.py::test_segment_explorer_loads_without_exception -v`
Expected: PASS

- [ ] **Step 5: Verify locally in browser**

Run: `streamlit run streamlit_app/app.py`
Open: http://localhost:8501 → navigate to Segment Explorer.
Expected: Both bar charts render. Heatmap renders with red Germany cells. Table shows 12 rows sorted by churn rate. Changing sidebar filters updates all 3 components. Stop with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/pages/1_Segment_Explorer.py streamlit_app/tests/test_pages.py
git commit -m "feat: add segment explorer page"
```

---

## Task 5: Model Insights

**Files:**
- Create: `streamlit_app/pages/2_Model_Insights.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `streamlit_app/tests/test_pages.py`:

```python
def test_model_insights_loads_without_exception():
    at = AppTest.from_file(str(_APP_DIR / "pages" / "2_Model_Insights.py"))
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest streamlit_app/tests/test_pages.py::test_model_insights_loads_without_exception -v`
Expected: `FileNotFoundError` — page not found

- [ ] **Step 3: Create 2_Model_Insights.py**

Create `streamlit_app/pages/2_Model_Insights.py`:

```python
from pathlib import Path
import streamlit as st
import plotly.express as px
from data_loader import load_lr_coefficients, load_feature_importance

st.set_page_config(page_title="Model Insights", layout="wide")
st.title("🤖 Model Insights")

st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("LR Accuracy", "86.3%")
col2.metric("LR ROC-AUC", "0.869")
col3.metric("DT Accuracy", "82.6%")
col4.metric("DT ROC-AUC", "0.830")

st.markdown("---")
st.subheader("Feature Importance")

lr_df = load_lr_coefficients()
dt_df = load_feature_importance()

col1, col2 = st.columns(2)

with col1:
    fig_lr = px.bar(
        lr_df.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        orientation="h",
        title="Logistic Regression Coefficients",
        labels={"coefficient": "Coefficient", "feature": ""},
        color="coefficient",
        color_continuous_scale=["#2196F3", "#F44336"],
    )
    fig_lr.update_coloraxes(showscale=False)
    fig_lr.update_layout(height=400)
    st.plotly_chart(fig_lr, use_container_width=True)

with col2:
    fig_dt = px.bar(
        dt_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Decision Tree Feature Importances",
        labels={"importance": "Importance", "feature": ""},
        color_discrete_sequence=["#4caf50"],
    )
    fig_dt.update_xaxes(tickformat=".2%")
    fig_dt.update_layout(height=400)
    st.plotly_chart(fig_dt, use_container_width=True)

st.markdown("---")

_FIGURES = Path(__file__).parent.parent.parent / "outputs" / "figures"

col1, col2 = st.columns(2)
with col1:
    st.subheader("ROC Curves")
    st.image(str(_FIGURES / "03_roc_curves.png"), use_container_width=True)
with col2:
    st.subheader("Confusion Matrices")
    st.image(str(_FIGURES / "03_confusion_matrices.png"), use_container_width=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest streamlit_app/tests/test_pages.py::test_model_insights_loads_without_exception -v`
Expected: PASS

- [ ] **Step 5: Verify locally in browser**

Run: `streamlit run streamlit_app/app.py`
Open Model Insights page.
Expected: 4 metric tiles across the top. LR coefficients chart shows Complain as the tallest bar. DT importances shows Complain at ~99.7%. ROC and confusion matrix PNGs render. Stop with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app/pages/2_Model_Insights.py streamlit_app/tests/test_pages.py
git commit -m "feat: add model insights page"
```

---

## Task 6: Retention Playbook

**Files:**
- Create: `streamlit_app/pages/3_Retention_Playbook.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `streamlit_app/tests/test_pages.py`:

```python
def test_retention_playbook_loads_without_exception():
    at = AppTest.from_file(str(_APP_DIR / "pages" / "3_Retention_Playbook.py"))
    at.run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest streamlit_app/tests/test_pages.py::test_retention_playbook_loads_without_exception -v`
Expected: `FileNotFoundError` — page not found

- [ ] **Step 3: Create 3_Retention_Playbook.py**

Create `streamlit_app/pages/3_Retention_Playbook.py`:

```python
import streamlit as st
import plotly.express as px
from data_loader import load_cross_segment_summary, load_retention_playbook

st.set_page_config(page_title="Retention Playbook", layout="wide")
st.title("📋 Retention Playbook")

cross_df = load_cross_segment_summary()
top5 = cross_df.nlargest(5, "churn_rate").copy()
top5["Segment"] = top5["Card Type"] + " · " + top5["Geography"]

st.subheader("Top 5 Highest-Risk Segments")
fig = px.bar(
    top5.sort_values("churn_rate"),
    x="churn_rate",
    y="Segment",
    orientation="h",
    title="Churn Rate — Top 5 Risk Segments",
    labels={"churn_rate": "Churn Rate", "Segment": ""},
    color="churn_rate",
    color_continuous_scale="Reds",
)
fig.update_xaxes(tickformat=".0%")
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Strategic Retention Themes")

playbook_df = load_retention_playbook()

THEME_ICONS = {
    "High Complaint Rate": "🚨",
    "Inactive Members": "😴",
    "Customers with 3+ Products": "📦",
    "GOLD & SILVER Cards in Germany": "🇩🇪",
    "Low Satisfaction Score": "⭐",
}

for theme, group in playbook_df.groupby("Theme", sort=False):
    icon = THEME_ICONS.get(theme, "📌")
    key_driver = group["Key Driver"].iloc[0]
    target = group["Target Segments"].iloc[0]
    actions = group["Recommended Action"].tolist()

    with st.expander(f"{icon} {theme}", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Key Driver**\n\n{key_driver}")
            st.markdown(f"**Target Segment**\n\n{target}")
        with col2:
            st.markdown("**Recommended Actions**")
            for action in actions:
                st.markdown(f"- {action}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest streamlit_app/tests/test_pages.py::test_retention_playbook_loads_without_exception -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `pytest streamlit_app/tests/ -v`
Expected: All 13 tests PASS (9 data loader + 4 page smoke tests)

- [ ] **Step 6: Verify locally in browser**

Run: `streamlit run streamlit_app/app.py`
Open Retention Playbook page.
Expected: Top 5 bar chart shows 4 Germany segments + Diamond France. 5 expandable strategy cards each with key driver, target, and 2 action bullets. Stop with Ctrl+C.

- [ ] **Step 7: Commit**

```bash
git add streamlit_app/pages/3_Retention_Playbook.py streamlit_app/tests/test_pages.py
git commit -m "feat: add retention playbook page"
```

---

## Task 7: Streamlit Cloud Deployment Prep

**Files:**
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Create Streamlit config**

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#ff4b4b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
```

- [ ] **Step 2: Final full-app smoke test**

Run: `streamlit run streamlit_app/app.py`
Navigate through all 4 pages in the browser:
- Home: 3 metric cards + 3 nav cards visible
- Segment Explorer: filters work, heatmap renders, table shows 12 rows
- Model Insights: 4 metric tiles + 2 Plotly charts + 2 PNGs
- Retention Playbook: top 5 bar chart + 5 strategy cards
Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "chore: add streamlit cloud config"
```

- [ ] **Step 4: Push to GitHub**

Run: `git push origin main`

- [ ] **Step 5: Deploy to Streamlit Cloud**

1. Go to https://share.streamlit.io
2. Click "New app"
3. Set: Repository = `alieissa81/customer-churn-analysis`, Branch = `main`, Main file path = `streamlit_app/app.py`
4. Click Deploy
5. Wait for the build to complete (2–3 minutes)
6. Verify all 4 pages work on the live URL
