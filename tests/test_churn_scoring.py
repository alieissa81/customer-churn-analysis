import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from skills.utils.churn_scoring import score_customers, top_churn_features


@pytest.fixture
def trained_lr():
    features = ['feature_a', 'feature_b']
    X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], columns=features)
    y = np.array([0, 0, 1, 1, 0, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, features


def test_score_customers_returns_dataframe(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6]], columns=features)
    result = score_customers(model, X_test)
    assert isinstance(result, pd.DataFrame)
    assert 'churn_probability' in result.columns


def test_score_customers_probabilities_in_range(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6], [3, 3]], columns=features)
    result = score_customers(model, X_test)
    assert all(0.0 <= p <= 1.0 for p in result['churn_probability'])


def test_score_customers_row_count_matches(trained_lr):
    model, features = trained_lr
    X_test = pd.DataFrame([[1, 2], [5, 6], [3, 3]], columns=features)
    result = score_customers(model, X_test)
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
