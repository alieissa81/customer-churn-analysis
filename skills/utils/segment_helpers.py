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
    return df[mask].copy()
