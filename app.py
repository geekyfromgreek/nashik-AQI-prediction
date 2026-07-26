import streamlit as st
from utils import inject_css, render_prediction_sidebar, load_raw_data, render_kpi_card

st.set_page_config(
    page_title="Air Quality Analysis — Nashik",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

st.markdown(
    '<div class="page-header">'
    "<h1>Air Quality Analysis for Clean Air &amp; Pollution Prevention</h1>"
    "<p>Nashik City — AQI Monitoring, Trend Analysis &amp; Prediction Dashboard</p>"
    "</div>",
    unsafe_allow_html=True,
)

raw_df = load_raw_data()

cards_html = '<div class="kpi-row">'
cards_html += render_kpi_card(
    "Total Records",
    f"{len(raw_df):,}",
    subtitle="Data points collected",
    color="#2E7D32",
)
cards_html += render_kpi_card(
    "Date Range",
    f"{raw_df['date'].dt.year.min()} – {raw_df['date'].dt.year.max()}",
    subtitle="Years of monitoring",
    color="#1565C0",
)
cards_html += render_kpi_card(
    "Average AQI",
    f"{raw_df['Index Value'].mean():.1f}",
    subtitle="Overall mean Index Value",
    color="#EF6C00",
)
cards_html += render_kpi_card(
    "Pollutant Types",
    str(raw_df["Prominent Pollutant"].nunique()),
    subtitle="Distinct prominent pollutants",
    color="#7B1FA2",
)
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

st.markdown("## Explore the Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Dashboard</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Key performance indicators including predicted AQI, "
        "average / max / min values, and dataset summary metrics."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Data Explorer</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Browse the raw dataset, review column types, check missing values, "
        "and view descriptive statistics."
        "</p></div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Visualizations</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Interactive Plotly charts — AQI trends, monthly patterns, "
        "distributions, pollutant analysis, and correlation heatmaps."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">AQI Prediction</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Predict air quality using the trained ANN model. Enter parameters "
        "in the sidebar and view predictions with visual gauges."
        "</p></div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">Model Performance</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Compare 9 regression models on MAE, RMSE and R² Score. "
        "See why ANN was selected as the final model."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-card">'
        '<div class="section-card-header">About Project</div>'
        "<p style='color:#64748B;font-size:13px;line-height:1.6;'>"
        "Project overview, objectives, methodology, ANN architecture, "
        "feature engineering details, and future scope."
        "</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:12px;'>"
    "B.E. Artificial Intelligence &amp; Data Science — Air Quality Analysis Project"
    "</p>",
    unsafe_allow_html=True,
)
