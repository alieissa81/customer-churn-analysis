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
