import pytest
import pandas as pd
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
