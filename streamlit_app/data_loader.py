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
