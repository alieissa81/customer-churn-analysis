import pandas as pd
import numpy as np


def score_customers(model, X):
    """Return DataFrame with churn probability per row."""
    probs = model.predict_proba(X)[:, 1]
    index = X.index if hasattr(X, 'index') else None
    return pd.DataFrame({'churn_probability': probs}, index=index)


def top_churn_features(model, feature_names, n=5):
    """Return top-N features by absolute coefficient magnitude (LogisticRegression)."""
    coefs = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    return coefs.sort_values(ascending=False).head(n)
