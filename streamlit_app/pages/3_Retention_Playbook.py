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
