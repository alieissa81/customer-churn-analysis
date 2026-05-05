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
