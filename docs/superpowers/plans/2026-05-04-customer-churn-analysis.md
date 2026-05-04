# Customer Churn Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-notebook Jupyter project that analyzes 10,000 customer churn records by card type and country, trains an interpretable prediction model, and produces a strategic retention playbook — structured for GitHub publishing.

**Architecture:** Four focused notebooks (EDA → Segment Analysis → Prediction Model → Recommendations) share three Python utility modules in `skills/utils/`. Notebooks read from the root CSV and write charts/tables to `outputs/`. All visualizations use matplotlib/seaborn for native GitHub rendering.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, nbformat, pytest

---

## File Map

| File | Created/Modified | Responsibility |
|------|-----------------|----------------|
| `requirements.txt` | Create | Pin all dependency versions |
| `.gitignore` | Create | Exclude outputs, checkpoints, pyc |
| `README.md` | Create | Project overview + ordered notebook links |
| `skills/__init__.py` | Create | Make skills a package |
| `skills/utils/__init__.py` | Create | Make utils a package |
| `skills/utils/segment_helpers.py` | Create | Segment filtering, churn rate, summary stats |
| `skills/utils/plotting.py` | Create | Standardized charts, color maps, savefig |
| `skills/utils/churn_scoring.py` | Create | Model scoring + feature importance helpers |
| `skills/superpowers/` | Create | Copied superpowers skill reference files |
| `tests/test_segment_helpers.py` | Create | pytest tests for segment_helpers |
| `tests/test_plotting.py` | Create | pytest tests for plotting |
| `tests/test_churn_scoring.py` | Create | pytest tests for churn_scoring |
| `notebooks/01_eda.ipynb` | Create | EDA: distributions, correlations, churn breakdowns |
| `notebooks/02_segment_analysis.ipynb` | Create | Churn by card type × country, segment summaries |
| `notebooks/03_prediction_model.ipynb` | Create | Logistic Regression + Decision Tree + evaluation |
| `notebooks/04_recommendations.ipynb` | Create | Top risk segments + strategic retention playbook |
| `outputs/figures/.gitkeep` | Create | Track figures directory in git |
| `outputs/tables/.gitkeep` | Create | Track tables directory in git |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `outputs/figures/.gitkeep`
- Create: `outputs/tables/.gitkeep`
- Create: `skills/__init__.py`
- Create: `skills/utils/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Install scikit-learn**

```bash
pip3 install scikit-learn
```

Expected: Successfully installed scikit-learn (version printed)

- [ ] **Step 2: Create requirements.txt**

Run this to capture pinned versions:

```bash
pip3 show pandas numpy matplotlib seaborn scikit-learn nbformat jupyter | grep -E "^(Name|Version)" | paste - - | awk '{print $2"=="$4}'
```

Write `requirements.txt` with the output, e.g.:

```
pandas==2.3.0
numpy==2.3.0
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==<installed-version>
nbformat==5.10.4
jupyter==<installed-version>
pytest==<installed-version>
```

Replace `<installed-version>` with actual output from the command above.

- [ ] **Step 3: Create .gitignore**

Write `.gitignore`:

```
# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Outputs (keep .gitkeep, ignore generated files)
outputs/figures/*.png
outputs/tables/*.csv

# Env
.env
.venv/
```

- [ ] **Step 4: Create output directories and package files**

```bash
touch "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/figures/.gitkeep"
touch "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/tables/.gitkeep"
touch "/Users/alieissa/Claude Code Folder/Customer Churn/skills/__init__.py"
touch "/Users/alieissa/Claude Code Folder/Customer Churn/skills/utils/__init__.py"
mkdir -p "/Users/alieissa/Claude Code Folder/Customer Churn/tests"
touch "/Users/alieissa/Claude Code Folder/Customer Churn/tests/__init__.py"
mkdir -p "/Users/alieissa/Claude Code Folder/Customer Churn/notebooks"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore outputs/ skills/__init__.py skills/utils/__init__.py tests/__init__.py notebooks/
git commit -m "chore: project scaffolding — directories, gitignore, requirements"
```

---

## Task 2: segment_helpers.py (TDD)

**Files:**
- Create: `skills/utils/segment_helpers.py`
- Create: `tests/test_segment_helpers.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_segment_helpers.py`:

```python
import pytest
import pandas as pd
import sys
sys.path.insert(0, '..')
from skills.utils.segment_helpers import compute_churn_rate, segment_summary, filter_segment


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Card Type': ['GOLD', 'GOLD', 'SILVER', 'SILVER', 'DIAMOND'],
        'Geography': ['France', 'Germany', 'France', 'France', 'Germany'],
        'Exited': [1, 0, 1, 0, 1],
        'Satisfaction Score': [2, 4, 3, 5, 1],
        'Balance': [1000.0, 2000.0, 500.0, 3000.0, 4000.0],
        'CreditScore': [600, 700, 650, 750, 500],
        'Complain': [1, 0, 1, 0, 1],
    })


def test_compute_churn_rate_single_column(sample_df):
    result = compute_churn_rate(sample_df, 'Card Type')
    gold_row = result[result['Card Type'] == 'GOLD'].iloc[0]
    assert gold_row['churn_rate'] == pytest.approx(0.5)
    assert gold_row['n_customers'] == 2


def test_compute_churn_rate_multiple_columns(sample_df):
    result = compute_churn_rate(sample_df, ['Card Type', 'Geography'])
    assert 'Card Type' in result.columns
    assert 'Geography' in result.columns
    assert 'churn_rate' in result.columns
    assert 'n_customers' in result.columns


def test_segment_summary_required_columns(sample_df):
    result = segment_summary(sample_df, 'Geography')
    for col in ['churn_rate', 'n_customers', 'avg_satisfaction', 'avg_balance', 'avg_credit_score', 'complaint_rate']:
        assert col in result.columns, f"Missing column: {col}"


def test_segment_summary_values(sample_df):
    result = segment_summary(sample_df, 'Geography')
    france_row = result[result['Geography'] == 'France'].iloc[0]
    assert france_row['n_customers'] == 3
    assert france_row['churn_rate'] == pytest.approx(2 / 3)


def test_filter_segment_by_card_type(sample_df):
    result = filter_segment(sample_df, card_type='GOLD')
    assert len(result) == 2
    assert all(result['Card Type'] == 'GOLD')


def test_filter_segment_by_country(sample_df):
    result = filter_segment(sample_df, geography='France')
    assert len(result) == 3
    assert all(result['Geography'] == 'France')


def test_filter_segment_combined(sample_df):
    result = filter_segment(sample_df, card_type='SILVER', geography='France')
    assert len(result) == 2
    assert all(result['Card Type'] == 'SILVER')
    assert all(result['Geography'] == 'France')


def test_filter_segment_no_args_returns_all(sample_df):
    result = filter_segment(sample_df)
    assert len(result) == len(sample_df)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/test_segment_helpers.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` — `segment_helpers` does not exist yet.

- [ ] **Step 3: Write the implementation**

Write `skills/utils/segment_helpers.py`:

```python
import pandas as pd


def compute_churn_rate(df, group_cols):
    """Returns churn rate and customer count per group."""
    return (
        df.groupby(group_cols)['Exited']
        .agg(churn_rate='mean', n_customers='count')
        .reset_index()
    )


def segment_summary(df, group_cols):
    """Returns churn rate + key metrics per segment."""
    return (
        df.groupby(group_cols)
        .agg(
            churn_rate=('Exited', 'mean'),
            n_customers=('Exited', 'count'),
            avg_satisfaction=('Satisfaction Score', 'mean'),
            avg_balance=('Balance', 'mean'),
            avg_credit_score=('CreditScore', 'mean'),
            complaint_rate=('Complain', 'mean'),
        )
        .reset_index()
    )


def filter_segment(df, card_type=None, geography=None):
    """Filter DataFrame by card type and/or geography."""
    mask = pd.Series([True] * len(df), index=df.index)
    if card_type is not None:
        mask &= df['Card Type'] == card_type
    if geography is not None:
        mask &= df['Geography'] == geography
    return df[mask]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/test_segment_helpers.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/utils/segment_helpers.py tests/test_segment_helpers.py
git commit -m "feat: add segment_helpers utility with TDD"
```

---

## Task 3: plotting.py (TDD)

**Files:**
- Create: `skills/utils/plotting.py`
- Create: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_plotting.py`:

```python
import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from skills.utils.plotting import plot_churn_bar, save_fig, CARD_COLORS, COUNTRY_COLORS


def test_card_colors_has_all_types():
    for card in ['DIAMOND', 'GOLD', 'PLATINUM', 'SILVER']:
        assert card in CARD_COLORS, f"Missing color for {card}"


def test_country_colors_has_all_countries():
    for country in ['France', 'Germany', 'Spain']:
        assert country in COUNTRY_COLORS, f"Missing color for {country}"


def test_plot_churn_bar_returns_figure():
    data = pd.DataFrame({'Card Type': ['GOLD', 'SILVER'], 'churn_rate': [0.25, 0.15]})
    fig = plot_churn_bar(data, 'Card Type', title='Test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_churn_bar_with_color_map():
    data = pd.DataFrame({'Card Type': ['GOLD', 'SILVER'], 'churn_rate': [0.25, 0.15]})
    fig = plot_churn_bar(data, 'Card Type', color_map=CARD_COLORS)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_churn_bar_unknown_key_uses_fallback():
    data = pd.DataFrame({'x': ['UNKNOWN'], 'churn_rate': [0.1]})
    fig = plot_churn_bar(data, 'x', color_map=CARD_COLORS)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_fig_creates_file(tmp_path):
    fig, _ = plt.subplots()
    save_fig(fig, 'test_chart.png', output_dir=str(tmp_path))
    assert (tmp_path / 'test_chart.png').exists()
    plt.close(fig)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/test_plotting.py -v 2>&1 | head -20
```

Expected: `ImportError` — `plotting` does not exist yet.

- [ ] **Step 3: Write the implementation**

Write `skills/utils/plotting.py`:

```python
import os
import matplotlib.pyplot as plt


CARD_COLORS = {
    'DIAMOND': '#1565C0',
    'GOLD': '#F9A825',
    'PLATINUM': '#757575',
    'SILVER': '#90A4AE',
}

COUNTRY_COLORS = {
    'France': '#1565C0',
    'Germany': '#B71C1C',
    'Spain': '#E65100',
}


def save_fig(fig, filename, output_dir='outputs/figures'):
    """Save figure to output_dir/filename, creating the directory if needed."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)


def plot_churn_bar(data, x_col, churn_col='churn_rate', title='', color_map=None):
    """Bar chart of churn rate by a categorical column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = (
        [color_map.get(v, '#607D8B') for v in data[x_col]]
        if color_map else '#607D8B'
    )
    ax.bar(data[x_col], data[churn_col] * 100, color=colors, edgecolor='white')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title(title)
    ax.set_ylim(0, min(data[churn_col].max() * 130, 100))
    return fig
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/test_plotting.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/utils/plotting.py tests/test_plotting.py
git commit -m "feat: add plotting utility with TDD"
```

---

## Task 4: churn_scoring.py (TDD)

**Files:**
- Create: `skills/utils/churn_scoring.py`
- Create: `tests/test_churn_scoring.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_churn_scoring.py`:

```python
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
from skills.utils.churn_scoring import score_customers, top_churn_features


@pytest.fixture
def trained_lr():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    y = np.array([0, 0, 1, 1, 0, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, ['feature_a', 'feature_b']


def test_score_customers_returns_dataframe(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6]], columns=features)
    result = score_customers(model, X_test, features)
    assert isinstance(result, pd.DataFrame)
    assert 'churn_probability' in result.columns


def test_score_customers_probabilities_in_range(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6], [3, 3]], columns=features)
    result = score_customers(model, X_test, features)
    assert all(0.0 <= p <= 1.0 for p in result['churn_probability'])


def test_score_customers_row_count_matches(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6], [3, 3]], columns=features)
    result = score_customers(model, X_test, features)
    assert len(result) == 3


def test_top_churn_features_returns_series(trained_lr):
    model, features = trained_lr
    result = top_churn_features(model, features, n=2)
    assert hasattr(result, 'index')
    assert len(result) == 2


def test_top_churn_features_sorted_descending(trained_lr):
    model, features = trained_lr
    result = top_churn_features(model, features, n=2)
    values = result.values.tolist()
    assert values == sorted(values, reverse=True)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/test_churn_scoring.py -v 2>&1 | head -20
```

Expected: `ImportError` — `churn_scoring` does not exist yet.

- [ ] **Step 3: Write the implementation**

Write `skills/utils/churn_scoring.py`:

```python
import pandas as pd
import numpy as np


def score_customers(model, X, feature_names):
    """Return DataFrame with churn probability per row."""
    probs = model.predict_proba(X)[:, 1]
    index = X.index if hasattr(X, 'index') else None
    return pd.DataFrame({'churn_probability': probs}, index=index)


def top_churn_features(model, feature_names, n=5):
    """Return top-N features by absolute coefficient magnitude (LogisticRegression)."""
    coefs = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    return coefs.sort_values(ascending=False).head(n)
```

- [ ] **Step 4: Run all tests — verify they pass**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 -m pytest tests/ -v
```

Expected: All tests across all three test files PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/utils/churn_scoring.py tests/test_churn_scoring.py
git commit -m "feat: add churn_scoring utility with TDD"
```

---

## Task 5: Copy Superpowers Skills

**Files:**
- Create: `skills/superpowers/` (directory with copied skill files)

- [ ] **Step 1: Copy relevant skill files**

```bash
SKILLS_SRC="/Users/alieissa/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.7/skills"
DEST="/Users/alieissa/Claude Code Folder/Customer Churn/skills/superpowers"
mkdir -p "$DEST"

for skill in brainstorming writing-plans executing-plans subagent-driven-development \
             verification-before-completion using-superpowers systematic-debugging \
             test-driven-development finishing-a-development-branch; do
    if [ -d "$SKILLS_SRC/$skill" ]; then
        cp -r "$SKILLS_SRC/$skill" "$DEST/$skill"
    elif [ -f "$SKILLS_SRC/$skill.md" ]; then
        cp "$SKILLS_SRC/$skill.md" "$DEST/$skill.md"
    fi
done

ls "$DEST"
```

Expected: List of copied skill directories/files printed.

- [ ] **Step 2: Commit**

```bash
git add "skills/superpowers/"
git commit -m "chore: copy superpowers skill references into skills/superpowers"
```

---

## Task 6: Notebook 01 — EDA

**Files:**
- Create: `notebooks/01_eda.ipynb`

- [ ] **Step 1: Create the notebook using nbformat**

Write `create_01_eda.py` in the project root:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

cells = []

cells.append(nbf.v4.new_markdown_cell("# Notebook 01 — Exploratory Data Analysis\n\nOverall data quality, distributions, and churn rate breakdowns."))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skills.utils.plotting import save_fig, CARD_COLORS, COUNTRY_COLORS
from skills.utils.segment_helpers import compute_churn_rate

sns.set_theme(style='whitegrid')
"""))

cells.append(nbf.v4.new_markdown_cell("## Load Data"))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('../Customer-Churn-Records.csv')
print(f"Shape: {df.shape}")
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("## Data Quality"))

cells.append(nbf.v4.new_code_cell("""\
print("=== Data Types ===")
print(df.dtypes)
print("\\n=== Missing Values ===")
print(df.isnull().sum())
assert df.isnull().sum().sum() == 0, "Unexpected nulls found"
print("\\nNo missing values.")
"""))

cells.append(nbf.v4.new_markdown_cell("## Overall Churn Rate"))

cells.append(nbf.v4.new_code_cell("""\
churn_rate = df['Exited'].mean()
n_churned = df['Exited'].sum()
print(f"Overall churn rate: {churn_rate:.1%}  ({n_churned} of {len(df)} customers)")
"""))

cells.append(nbf.v4.new_markdown_cell("## Feature Distributions"))

cells.append(nbf.v4.new_code_cell("""\
features = ['Age', 'Balance', 'CreditScore', 'Tenure', 'Point Earned', 'EstimatedSalary']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flatten(), features):
    df[feat].hist(ax=ax, bins=30, color='steelblue', edgecolor='white')
    ax.set_title(feat)
    ax.set_xlabel('')
plt.suptitle('Feature Distributions', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, '01_distributions.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Correlation Heatmap"))

cells.append(nbf.v4.new_code_cell("""\
numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited',
                'Complain', 'Satisfaction Score', 'Point Earned']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=13)
plt.tight_layout()
save_fig(fig, '01_correlation_heatmap.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Churn Rate by Categorical Features"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, ['Gender', 'IsActiveMember', 'NumOfProducts', 'HasCrCard']):
    churn_by = df.groupby(col)['Exited'].mean() * 100
    churn_by.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Churn Rate by {col}')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)
plt.suptitle('Churn Rate by Categorical Features', fontsize=14)
plt.tight_layout()
save_fig(fig, '01_churn_by_categorical.png')
plt.show()
"""))

nb.cells = cells

with open('notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Created notebooks/01_eda.ipynb")
```

- [ ] **Step 2: Run the creation script**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 create_01_eda.py
```

Expected: `Created notebooks/01_eda.ipynb`

- [ ] **Step 3: Execute the notebook to verify it runs end-to-end**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb --ExecutePreprocessor.timeout=120
```

Expected: `[NbConvertApp] Writing ... bytes to notebooks/01_eda.ipynb` with no errors.

- [ ] **Step 4: Verify output files were created**

```bash
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/figures/"
```

Expected: `01_distributions.png`, `01_correlation_heatmap.png`, `01_churn_by_categorical.png`

- [ ] **Step 5: Delete creation script and commit**

```bash
rm "/Users/alieissa/Claude Code Folder/Customer Churn/create_01_eda.py"
git add notebooks/01_eda.ipynb outputs/
git commit -m "feat: add EDA notebook (01_eda)"
```

---

## Task 7: Notebook 02 — Segment Analysis

**Files:**
- Create: `notebooks/02_segment_analysis.ipynb`

- [ ] **Step 1: Create the notebook**

Write `create_02_segment.py` in the project root:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

cells = []

cells.append(nbf.v4.new_markdown_cell("# Notebook 02 — Segment Analysis\n\nChurn rate and key metrics broken down by Card Type and Country, including cross-tab heatmap."))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skills.utils.plotting import save_fig, CARD_COLORS, COUNTRY_COLORS, plot_churn_bar
from skills.utils.segment_helpers import compute_churn_rate, segment_summary

sns.set_theme(style='whitegrid')
"""))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('../Customer-Churn-Records.csv')
print(f"Loaded {len(df)} records")
"""))

cells.append(nbf.v4.new_markdown_cell("## Churn Rate by Card Type"))

cells.append(nbf.v4.new_code_cell("""\
card_churn = compute_churn_rate(df, 'Card Type').sort_values('churn_rate', ascending=False)
fig = plot_churn_bar(card_churn, 'Card Type', title='Churn Rate by Card Type', color_map=CARD_COLORS)
save_fig(fig, '02_churn_by_card_type.png')
plt.show()
print(card_churn.to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## Churn Rate by Country"))

cells.append(nbf.v4.new_code_cell("""\
country_churn = compute_churn_rate(df, 'Geography').sort_values('churn_rate', ascending=False)
fig = plot_churn_bar(country_churn, 'Geography', title='Churn Rate by Country', color_map=COUNTRY_COLORS)
save_fig(fig, '02_churn_by_country.png')
plt.show()
print(country_churn.to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## Card Type × Country Churn Heatmap"))

cells.append(nbf.v4.new_code_cell("""\
pivot = df.groupby(['Card Type', 'Geography'])['Exited'].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Churn Rate (%)'})
ax.set_title('Churn Rate (%) by Card Type × Country', fontsize=13)
plt.tight_layout()
save_fig(fig, '02_churn_heatmap.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Segment Summary — by Card Type"))

cells.append(nbf.v4.new_code_cell("""\
card_summary = segment_summary(df, 'Card Type').sort_values('churn_rate', ascending=False)
card_summary['churn_rate'] = card_summary['churn_rate'].round(3)
card_summary['complaint_rate'] = card_summary['complaint_rate'].round(3)
print(card_summary.to_string(index=False))
card_summary.to_csv('../outputs/tables/card_type_summary.csv', index=False)
print("\\nSaved to outputs/tables/card_type_summary.csv")
"""))

cells.append(nbf.v4.new_markdown_cell("## Segment Summary — by Country"))

cells.append(nbf.v4.new_code_cell("""\
country_summary = segment_summary(df, 'Geography').sort_values('churn_rate', ascending=False)
country_summary['churn_rate'] = country_summary['churn_rate'].round(3)
country_summary['complaint_rate'] = country_summary['complaint_rate'].round(3)
print(country_summary.to_string(index=False))
country_summary.to_csv('../outputs/tables/country_summary.csv', index=False)
print("\\nSaved to outputs/tables/country_summary.csv")
"""))

cells.append(nbf.v4.new_markdown_cell("## Cross-Segment Summary — Card Type × Country"))

cells.append(nbf.v4.new_code_cell("""\
cross_summary = segment_summary(df, ['Card Type', 'Geography']).sort_values('churn_rate', ascending=False)
cross_summary['churn_rate'] = cross_summary['churn_rate'].round(3)
print("Top 10 highest-churn segments:")
print(cross_summary.head(10).to_string(index=False))
cross_summary.to_csv('../outputs/tables/cross_segment_summary.csv', index=False)
print("\\nSaved to outputs/tables/cross_segment_summary.csv")
"""))

nb.cells = cells

with open('notebooks/02_segment_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Created notebooks/02_segment_analysis.ipynb")
```

- [ ] **Step 2: Run the creation script**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 create_02_segment.py
```

- [ ] **Step 3: Execute the notebook**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && jupyter nbconvert --to notebook --execute notebooks/02_segment_analysis.ipynb --output notebooks/02_segment_analysis.ipynb --ExecutePreprocessor.timeout=120
```

Expected: No errors. Three figures + three CSVs created.

- [ ] **Step 4: Verify outputs**

```bash
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/figures/" | grep 02
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/tables/"
```

Expected: `02_churn_by_card_type.png`, `02_churn_by_country.png`, `02_churn_heatmap.png` and three CSVs.

- [ ] **Step 5: Delete creation script and commit**

```bash
rm "/Users/alieissa/Claude Code Folder/Customer Churn/create_02_segment.py"
git add notebooks/02_segment_analysis.ipynb outputs/
git commit -m "feat: add segment analysis notebook (02_segment_analysis)"
```

---

## Task 8: Notebook 03 — Prediction Model

**Files:**
- Create: `notebooks/03_prediction_model.ipynb`

- [ ] **Step 1: Create the notebook**

Write `create_03_model.py` in the project root:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

cells = []

cells.append(nbf.v4.new_markdown_cell("# Notebook 03 — Churn Prediction Model\n\nLogistic Regression and Decision Tree trained on customer features. Includes coefficients, tree visualization, confusion matrix, and ROC curves."))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              confusion_matrix, roc_auc_score, roc_curve,
                              ConfusionMatrixDisplay)
from skills.utils.plotting import save_fig
from skills.utils.churn_scoring import score_customers, top_churn_features

sns.set_theme(style='whitegrid')
"""))

cells.append(nbf.v4.new_markdown_cell("## Load & Prepare Features"))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('../Customer-Churn-Records.csv')
df_model = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
df_encoded = pd.get_dummies(df_model, columns=['Geography', 'Gender', 'Card Type'], drop_first=False)

X = df_encoded.drop(columns=['Exited'])
y = df_encoded['Exited']
feature_names = X.columns.tolist()

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"Class balance: {y.value_counts().to_dict()}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Train / Test Split"))

cells.append(nbf.v4.new_code_cell("""\
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Logistic Regression"))

cells.append(nbf.v4.new_code_cell("""\
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_lr):.3f}")
"""))

cells.append(nbf.v4.new_markdown_cell("### Logistic Regression Coefficients\n\nPositive coefficients increase churn risk; negative coefficients decrease it."))

cells.append(nbf.v4.new_code_cell("""\
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#E53935' if c > 0 else '#1E88E5' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Logistic Regression Coefficients\\n(red = increases churn risk, blue = decreases)', fontsize=12)
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
save_fig(fig, '03_lr_coefficients.png')
plt.show()

coef_df.to_csv('../outputs/tables/lr_coefficients.csv', index=False)
print(coef_df.to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## Decision Tree"))

cells.append(nbf.v4.new_code_cell("""\
dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print("=== Decision Tree Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_dt):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_dt):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_dt):.3f}")
"""))

cells.append(nbf.v4.new_markdown_cell("### Decision Tree Visualization"))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(dt, feature_names=feature_names, class_names=['Stay', 'Churn'],
          filled=True, rounded=True, ax=ax, fontsize=8, impurity=False)
ax.set_title('Decision Tree (max_depth=4)', fontsize=13)
save_fig(fig, '03_decision_tree.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("### Feature Importances (Decision Tree)"))

cells.append(nbf.v4.new_code_cell("""\
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(fi_df['feature'], fi_df['importance'], color='steelblue', edgecolor='white')
ax.set_title('Top 10 Feature Importances (Decision Tree)', fontsize=12)
ax.set_xlabel('Importance')
plt.tight_layout()
save_fig(fig, '03_feature_importance.png')
plt.show()

fi_df.to_csv('../outputs/tables/feature_importance.csv', index=False)
print(fi_df.to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## Confusion Matrices"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name, y_pred in zip(axes,
                             ['Logistic Regression', 'Decision Tree'],
                             [y_pred_lr, y_pred_dt]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Stay', 'Churn'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name, fontsize=12)
plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
save_fig(fig, '03_confusion_matrices.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## ROC Curves"))

cells.append(nbf.v4.new_code_cell("""\
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_dt = roc_auc_score(y_test, y_prob_dt)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2)
ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.3f})', linewidth=2, linestyle='--')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Churn Prediction Models', fontsize=12)
ax.legend()
plt.tight_layout()
save_fig(fig, '03_roc_curves.png')
plt.show()
"""))

nb.cells = cells

with open('notebooks/03_prediction_model.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Created notebooks/03_prediction_model.ipynb")
```

- [ ] **Step 2: Run the creation script**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 create_03_model.py
```

- [ ] **Step 3: Execute the notebook**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && jupyter nbconvert --to notebook --execute notebooks/03_prediction_model.ipynb --output notebooks/03_prediction_model.ipynb --ExecutePreprocessor.timeout=180
```

Expected: No errors. Figures and tables saved.

- [ ] **Step 4: Verify outputs**

```bash
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/figures/" | grep 03
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/tables/"
```

Expected: `03_lr_coefficients.png`, `03_decision_tree.png`, `03_feature_importance.png`, `03_confusion_matrices.png`, `03_roc_curves.png` and `lr_coefficients.csv`, `feature_importance.csv`.

- [ ] **Step 5: Delete creation script and commit**

```bash
rm "/Users/alieissa/Claude Code Folder/Customer Churn/create_03_model.py"
git add notebooks/03_prediction_model.ipynb outputs/
git commit -m "feat: add prediction model notebook (03_prediction_model)"
```

---

## Task 9: Notebook 04 — Strategic Recommendations

**Files:**
- Create: `notebooks/04_recommendations.ipynb`

- [ ] **Step 1: Create the notebook**

Write `create_04_recommendations.py` in the project root:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

cells = []

cells.append(nbf.v4.new_markdown_cell("# Notebook 04 — Strategic Retention Playbook\n\nIdentifies the highest-risk customer segments and maps each to concrete retention actions."))

cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from skills.utils.plotting import save_fig
from skills.utils.segment_helpers import compute_churn_rate, segment_summary

sns.set_theme(style='whitegrid')
"""))

cells.append(nbf.v4.new_markdown_cell("## Load Data & Pre-computed Results"))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('../Customer-Churn-Records.csv')
cross_summary = pd.read_csv('../outputs/tables/cross_segment_summary.csv')
lr_coef = pd.read_csv('../outputs/tables/lr_coefficients.csv')
fi = pd.read_csv('../outputs/tables/feature_importance.csv')
overall_churn = df['Exited'].mean()
print(f"Overall churn rate: {overall_churn:.1%}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Top 5 Highest-Risk Segments"))

cells.append(nbf.v4.new_code_cell("""\
top_segments = cross_summary.sort_values('churn_rate', ascending=False).head(5)
print("Top 5 Highest-Risk Segments:")
display_cols = ['Card Type', 'Geography', 'churn_rate', 'n_customers', 'avg_satisfaction', 'complaint_rate']
print(top_segments[display_cols].to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(11, 6))
labels = top_segments['Card Type'] + '\\n(' + top_segments['Geography'] + ')'
bars = ax.bar(labels, top_segments['churn_rate'] * 100, color='#E53935', edgecolor='white')
ax.axhline(overall_churn * 100, color='steelblue', linestyle='--', linewidth=1.5,
           label=f'Overall avg ({overall_churn:.1%})')
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Top 5 Highest-Risk Customer Segments', fontsize=13)
ax.legend()
for bar, val in zip(bars, top_segments['churn_rate'] * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
save_fig(fig, '04_top_risk_segments.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Top Churn Drivers (from Logistic Regression)"))

cells.append(nbf.v4.new_code_cell("""\
top_positive = lr_coef[lr_coef['coefficient'] > 0].sort_values('coefficient', ascending=False).head(5)
top_negative = lr_coef[lr_coef['coefficient'] < 0].sort_values('coefficient').head(5)
print("Top factors INCREASING churn risk:")
print(top_positive[['feature', 'coefficient']].to_string(index=False))
print("\\nTop factors DECREASING churn risk (protective):")
print(top_negative[['feature', 'coefficient']].to_string(index=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## Strategic Retention Playbook"))

cells.append(nbf.v4.new_code_cell("""\
recommendations = [
    {
        'Theme': 'High Complaint Rate',
        'Key Driver': 'Complain (strongest churn predictor in LR)',
        'Target Segments': 'All segments — especially SILVER/GOLD in Germany',
        'Actions': [
            'Launch proactive outreach for all customers with any complaint history: assign a dedicated relationship manager for 30 days after a complaint.',
            'Implement a complaint-resolution SLA (48h escalation) + automatic follow-up with satisfaction survey and a goodwill gesture (fee waiver or bonus points).',
        ]
    },
    {
        'Theme': 'Inactive Members',
        'Key Driver': 'IsActiveMember = 0 (strong negative predictor)',
        'Target Segments': 'Any card type with >60 days of inactivity',
        'Actions': [
            'Trigger a re-engagement email + app push notification after 60 days of inactivity, offering a 2x points multiplier for the next 30 days.',
            'For inactive high-balance customers, offer a free annual financial health review to surface better-fit products.',
        ]
    },
    {
        'Theme': 'Customers with 3+ Products',
        'Key Driver': 'NumOfProducts >= 3 shows high churn in EDA',
        'Target Segments': 'Any segment where NumOfProducts = 3 or 4',
        'Actions': [
            'Audit product bundling — customers with 3+ products may feel over-sold; offer a free annual portfolio review to right-size.',
            'Create a loyalty tier benefit for multi-product holders (priority service, higher savings rates) to increase perceived value.',
        ]
    },
    {
        'Theme': 'GOLD & SILVER Cards in Germany',
        'Key Driver': 'Highest churn rate + complaint rate in Germany cross-segment',
        'Target Segments': 'GOLD Germany, SILVER Germany',
        'Actions': [
            'Run a targeted card upgrade campaign (GOLD→PLATINUM or SILVER→GOLD) with reduced annual fee for the first year.',
            'Survey recent churners in Germany to surface market-specific pain points and address top 3 issues within one quarter.',
        ]
    },
    {
        'Theme': 'Low Satisfaction Score',
        'Key Driver': 'Satisfaction Score (protective when high, risky when low)',
        'Target Segments': 'Any segment with avg satisfaction score < 3',
        'Actions': [
            'Implement quarterly NPS surveys; flag customers scoring ≤6 for immediate retention team follow-up.',
            'Create a Satisfaction Guarantee: customers dissatisfied with a product in first 90 days get a no-penalty switch to an alternative product.',
        ]
    },
]

for rec in recommendations:
    print(f"\\n{'='*65}")
    print(f"THEME:          {rec['Theme']}")
    print(f"KEY DRIVER:     {rec['Key Driver']}")
    print(f"TARGET:         {rec['Target Segments']}")
    print(f"ACTIONS:")
    for i, action in enumerate(rec['Actions'], 1):
        print(f"  {i}. {action}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Export Playbook Table"))

cells.append(nbf.v4.new_code_cell("""\
rows = []
for rec in recommendations:
    for action in rec['Actions']:
        rows.append({
            'Theme': rec['Theme'],
            'Key Driver': rec['Key Driver'],
            'Target Segments': rec['Target Segments'],
            'Recommended Action': action,
        })
playbook_df = pd.DataFrame(rows)
playbook_df.to_csv('../outputs/tables/retention_playbook.csv', index=False)
print(f"Retention playbook saved: {len(playbook_df)} action rows")
print(playbook_df[['Theme', 'Target Segments']].to_string(index=False))
"""))

nb.cells = cells

with open('notebooks/04_recommendations.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Created notebooks/04_recommendations.ipynb")
```

- [ ] **Step 2: Run the creation script**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && python3 create_04_recommendations.py
```

- [ ] **Step 3: Execute the notebook**

```bash
cd "/Users/alieissa/Claude Code Folder/Customer Churn" && jupyter nbconvert --to notebook --execute notebooks/04_recommendations.ipynb --output notebooks/04_recommendations.ipynb --ExecutePreprocessor.timeout=120
```

Expected: No errors. `04_top_risk_segments.png` and `retention_playbook.csv` created.

- [ ] **Step 4: Verify outputs**

```bash
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/figures/" | grep 04
ls "/Users/alieissa/Claude Code Folder/Customer Churn/outputs/tables/"
```

Expected: `04_top_risk_segments.png` and `retention_playbook.csv`.

- [ ] **Step 5: Delete creation script and commit**

```bash
rm "/Users/alieissa/Claude Code Folder/Customer Churn/create_04_recommendations.py"
git add notebooks/04_recommendations.ipynb outputs/
git commit -m "feat: add recommendations notebook (04_recommendations)"
```

---

## Task 10: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

Write `README.md`:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README with notebook index and setup instructions"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|------------|
| Folder structure | Task 1 |
| Analysis by card type | Task 7 (02_segment_analysis) |
| Analysis by country | Task 7 (02_segment_analysis) |
| Churn prediction model | Task 8 (03_prediction_model) |
| Logistic Regression coefficients | Task 8 |
| Decision Tree visualization | Task 8 |
| Model evaluation metrics | Task 8 |
| Strategic recommendations per segment | Task 9 (04_recommendations) |
| skills/utils Python modules | Tasks 2–4 |
| skills/superpowers copies | Task 5 |
| requirements.txt | Task 1 |
| README.md | Task 10 |
| GitHub-renderable visualizations (matplotlib/seaborn only) | All notebooks |
| outputs/figures + outputs/tables directories | Task 1 |

All spec requirements have a corresponding task. No gaps found.

**Placeholder scan:** No TBD, TODO, or incomplete steps. All code blocks are complete.

**Type consistency:** `compute_churn_rate`, `segment_summary`, `filter_segment`, `score_customers`, `top_churn_features`, `save_fig`, `plot_churn_bar`, `CARD_COLORS`, `COUNTRY_COLORS` — all defined in Tasks 2–4 and used consistently in Tasks 6–9.
