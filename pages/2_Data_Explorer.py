import pandas as pd
import streamlit as st
from utils import (
    inject_css,
    load_processed_data,
    load_raw_data,
    render_page_header,
    render_prediction_sidebar,
)

st.set_page_config(
    page_title="Data Explorer — AQI Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

render_page_header(
    "Data Explorer",
    "Browse the dataset, review column information, and inspect data quality.",
)

raw_df = load_raw_data()
processed_df = load_processed_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Preview", 
    "Dataset Info", 
    "Missing Values", 
    "Summary Statistics", 
    "Feature List"
])

with tab1:
    st.markdown("## Raw Dataset")
    st.markdown(
        f'<p class="page-subheader">'
        f"Showing {len(raw_df):,} records with {len(raw_df.columns)} columns. "
        f"Date range: {raw_df['date'].min().strftime('%Y-%m-%d')} to "
        f"{raw_df['date'].max().strftime('%Y-%m-%d')}."
        f"</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(raw_df, use_container_width=True, height=420)

    st.markdown("---")
    st.markdown("## Processed Dataset (Feature-Engineered)")
    st.markdown(
        f'<p class="page-subheader">'
        f"{len(processed_df):,} records, {len(processed_df.columns)} columns "
        f"after feature extraction and one-hot encoding."
        f"</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(processed_df, use_container_width=True, height=420)

with tab2:
    st.markdown("## Column Information")

    info_rows = []
    for col in raw_df.columns:
        info_rows.append({
            "Column": col,
            "Data Type": str(raw_df[col].dtype),
            "Non-Null Count": int(raw_df[col].notna().sum()),
            "Null Count": int(raw_df[col].isna().sum()),
            "Unique Values": int(raw_df[col].nunique()),
        })
    info_df = pd.DataFrame(info_rows)
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(raw_df):,}")
    c2.metric("Total Columns", str(len(raw_df.columns)))
    c3.metric("Duplicate Rows", str(raw_df.duplicated().sum()))

with tab3:
    st.markdown("## Missing Values Analysis")

    missing = raw_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(raw_df) * 100).round(2)
    missing = missing.sort_values("Missing Count", ascending=False)

    total_missing = missing["Missing Count"].sum()
    if total_missing == 0:
        st.markdown(
            '<div class="model-highlight">'
            "<h4>No Missing Values</h4>"
            "<p>The raw dataset has zero null entries across all columns.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(missing, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Processed Dataset — Missing Values")
    missing_p = processed_df.isnull().sum().reset_index()
    missing_p.columns = ["Column", "Missing Count"]
    missing_p = missing_p[missing_p["Missing Count"] > 0]
    if missing_p.empty:
        st.info("No missing values in the processed dataset.")
    else:
        st.dataframe(missing_p, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("## Descriptive Statistics")
    st.markdown(
        '<p class="page-subheader">'
        "Statistical summary of the raw dataset's numeric columns."
        "</p>",
        unsafe_allow_html=True,
    )
    desc = raw_df.describe().T
    desc.index.name = "Column"
    st.dataframe(desc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Processed Dataset Statistics")
    desc_p = processed_df.describe().T
    desc_p.index.name = "Column"
    st.dataframe(desc_p, use_container_width=True)

with tab5:
    st.markdown("## Feature Descriptions")

    features_info = [
        ("No. Stations", "Number of monitoring stations", "Numeric"),
        ("year", "Year extracted from date", "Numeric"),
        ("day", "Day of month extracted from date", "Numeric"),
        ("month", "Month extracted from date", "Numeric"),
        ("weekdays", "Day of week (0 = Monday ... 6 = Sunday)", "Numeric"),
        ("month_sin", "Cyclic sine encoding of month", "Numeric"),
        ("month_cos", "Cyclic cosine encoding of month", "Numeric"),
        ("AQI_lag_1", "AQI value from 1 day ago", "Numeric"),
        ("AQI_lag_week_ago", "AQI value from 7 days ago", "Numeric"),
        ("Prominent Pollutant_*", "One-hot encoded pollutant (10 columns, CO is reference)", "Binary"),
        ("Index Value", "Target variable — AQI index value", "Numeric (Target)"),
        ("Air Quality", "AQI category label (not used as feature)", "Categorical"),
    ]
    feat_df = pd.DataFrame(features_info, columns=["Feature", "Description", "Type"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
