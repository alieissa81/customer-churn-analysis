import streamlit as st
import plotly.express as px
from data_loader import (
    load_card_type_summary,
    load_country_summary,
    load_cross_segment_summary,
)

st.set_page_config(page_title="Segment Explorer", layout="wide")
st.title("📊 Segment Explorer")

CARD_COLORS = {
    "DIAMOND": "#4e79a7",
    "GOLD": "#f28e2b",
    "PLATINUM": "#76b7b2",
    "SILVER": "#e15759",
}
COUNTRY_COLORS = {
    "Germany": "#e15759",
    "Spain": "#76b7b2",
    "France": "#4e79a7",
}

with st.sidebar:
    st.header("Filters")
    all_cards = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
    all_countries = ["France", "Germany", "Spain"]
    selected_cards = st.multiselect("Card Type", all_cards, default=all_cards)
    selected_countries = st.multiselect("Country", all_countries, default=all_countries)

if not selected_cards:
    selected_cards = all_cards
if not selected_countries:
    selected_countries = all_countries

card_df = load_card_type_summary()
country_df = load_country_summary()
cross_df = load_cross_segment_summary()

card_filtered = card_df[card_df["Card Type"].isin(selected_cards)].copy()
country_filtered = country_df[country_df["Geography"].isin(selected_countries)].copy()
cross_filtered = cross_df[
    cross_df["Card Type"].isin(selected_cards) & cross_df["Geography"].isin(selected_countries)
].copy()

col1, col2 = st.columns(2)

with col1:
    if not card_filtered.empty:
        fig = px.bar(
            card_filtered.sort_values("churn_rate"),
            x="churn_rate",
            y="Card Type",
            orientation="h",
            title="Churn Rate by Card Type",
            labels={"churn_rate": "Churn Rate", "Card Type": ""},
            color="Card Type",
            color_discrete_map=CARD_COLORS,
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

with col2:
    if not country_filtered.empty:
        fig2 = px.bar(
            country_filtered.sort_values("churn_rate"),
            x="churn_rate",
            y="Geography",
            orientation="h",
            title="Churn Rate by Country",
            labels={"churn_rate": "Churn Rate", "Geography": ""},
            color="Geography",
            color_discrete_map=COUNTRY_COLORS,
        )
        fig2.update_layout(showlegend=False)
        fig2.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

st.subheader("Churn Rate Heatmap — Card Type × Country")
if not cross_filtered.empty:
    pivot = cross_filtered.pivot(index="Card Type", columns="Geography", values="churn_rate")
    fig3 = px.imshow(
        pivot,
        text_auto=".1%",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Churn Rate by Segment",
    )
    fig3.update_coloraxes(colorbar_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data for the selected filters.")

st.subheader("Segment Summary")
if not cross_filtered.empty:
    display = cross_filtered.sort_values("churn_rate", ascending=False)[
        ["Card Type", "Geography", "n_customers", "churn_rate", "avg_balance", "complaint_rate"]
    ].copy()
    display = display.rename(columns={
        "Geography": "Country",
        "n_customers": "Customers",
        "churn_rate": "Churn Rate",
        "avg_balance": "Avg Balance",
        "complaint_rate": "Complaint Rate",
    })
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Avg Balance": st.column_config.NumberColumn(format="€%.0f"),
            "Complaint Rate": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
else:
    st.info("No data for the selected filters.")
