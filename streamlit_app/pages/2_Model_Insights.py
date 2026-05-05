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
st.subheader("Coefficients & Feature Importance")

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
