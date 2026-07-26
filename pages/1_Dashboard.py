import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    inject_css,
    load_raw_data,
    render_kpi_card,
    render_page_header,
    render_prediction_sidebar,
    plotly_layout,
)

st.set_page_config(
    page_title="Dashboard — AQI Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

render_page_header(
    "Dashboard",
    "Key performance indicators and air quality summary metrics.",
)

raw_df = load_raw_data()

avg_aqi = raw_df["Index Value"].mean()
max_aqi = raw_df["Index Value"].max()
min_aqi = raw_df["Index Value"].min()
total_records = len(raw_df)
n_cities = raw_df["City"].nunique()
n_pollutants = raw_df["Prominent Pollutant"].nunique()

if "predicted_aqi" in st.session_state:
    pred_aqi = st.session_state["predicted_aqi"]
    pred_cat = st.session_state["predicted_category"]
    pred_clr = st.session_state["predicted_color"]
else:
    pred_aqi = None
    pred_cat = "—"
    pred_clr = "#94A3B8"

row1 = '<div class="kpi-row">'
row1 += render_kpi_card(
    "Predicted AQI",
    f"{pred_aqi:.1f}" if pred_aqi is not None else "—",
    subtitle="Latest sidebar prediction",
    color=pred_clr,
)
row1 += render_kpi_card(
    "Air Quality Category",
    pred_cat,
    subtitle="CPCB classification",
    color=pred_clr,
)
row1 += render_kpi_card(
    "Average AQI",
    f"{avg_aqi:.1f}",
    subtitle="Dataset overall mean",
    color="#1565C0",
)
row1 += render_kpi_card(
    "Maximum AQI",
    str(max_aqi),
    subtitle="Highest recorded value",
    color="#D32F2F",
)
row1 += "</div>"
st.markdown(row1, unsafe_allow_html=True)

row2 = '<div class="kpi-row">'
row2 += render_kpi_card(
    "Minimum AQI",
    str(min_aqi),
    subtitle="Lowest recorded value",
    color="#388E3C",
)
row2 += render_kpi_card(
    "Total Records",
    f"{total_records:,}",
    subtitle="Data points in dataset",
    color="#2E7D32",
)
row2 += render_kpi_card(
    "Cities Monitored",
    f"Nashik ({n_cities})",
    subtitle="Unique city names",
    color="#5E35B1",
)
row2 += render_kpi_card(
    "Pollutant Types",
    str(n_pollutants),
    subtitle="Distinct prominent pollutants",
    color="#EF6C00",
)
row2 += "</div>"
st.markdown(row2, unsafe_allow_html=True)

st.markdown("## Quick Insights")

col_left, col_right = st.columns(2)

with col_left:
    yearly = raw_df.groupby(raw_df["date"].dt.year)["Index Value"].mean().reset_index()
    yearly.columns = ["Year", "Average AQI"]
    fig = px.line(
        yearly,
        x="Year",
        y="Average AQI",
        markers=True,
    )
    fig.update_traces(
        line=dict(color="#2E7D32", width=2.5),
        marker=dict(size=7, color="#2E7D32"),
    )
    fig.update_layout(**plotly_layout("Yearly Average AQI Trend", height=340))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    cat_counts = raw_df["Air Quality"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_counts["Category"] = pd.Categorical(
        cat_counts["Category"], categories=order, ordered=True
    )
    cat_counts = cat_counts.sort_values("Category")
    colors_map = {
        "Good": "#388E3C",
        "Satisfactory": "#689F38",
        "Moderate": "#F9A825",
        "Poor": "#EF6C00",
        "Very Poor": "#D32F2F",
        "Severe": "#7B1FA2",
    }
    fig2 = px.pie(
        cat_counts,
        names="Category",
        values="Count",
        color="Category",
        color_discrete_map=colors_map,
        hole=0.45,
    )
    fig2.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=11,
    )
    fig2.update_layout(**plotly_layout("Air Quality Category Distribution", height=340))
    st.plotly_chart(fig2, use_container_width=True)
