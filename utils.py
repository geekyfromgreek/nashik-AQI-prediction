import os
import sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup TensorFlow path for Windows long path compatibility
_TF_PKG = r"C:\tf_pkg"
if os.path.isdir(_TF_PKG) and _TF_PKG not in sys.path:
    sys.path.insert(0, _TF_PKG)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "app", "aqi_prediction_ann.weights.h5")
SCALER_PATH = os.path.join(BASE_DIR, "app", "scaler.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "Nashik_AQIBulletins.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed_aqi_data.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")

FEATURE_COLUMNS = [
    "No. Stations",
    "year",
    "day",
    "month",
    "weekdays",
    "month_sin",
    "month_cos",
    "AQI_lag_1",
    "AQI_lag_week_ago",
    "Prominent Pollutant_CO, PM10",
    "Prominent Pollutant_CO, PM2.5",
    "Prominent Pollutant_NO2",
    "Prominent Pollutant_O3",
    "Prominent Pollutant_O3, PM10",
    "Prominent Pollutant_O3, PM2.5",
    "Prominent Pollutant_OZONE",
    "Prominent Pollutant_PM10",
    "Prominent Pollutant_PM10, PM2.5",
    "Prominent Pollutant_PM2.5",
]

POLLUTANT_CATEGORIES = [
    "CO",
    "CO, PM10",
    "CO, PM2.5",
    "NO2",
    "O3",
    "O3, PM10",
    "O3, PM2.5",
    "OZONE",
    "PM10",
    "PM10, PM2.5",
    "PM2.5",
]

AQI_CATEGORIES = [
    {"name": "Good",         "min": 0,   "max": 50,  "color": "#388E3C"},
    {"name": "Satisfactory", "min": 51,  "max": 100, "color": "#689F38"},
    {"name": "Moderate",     "min": 101, "max": 200, "color": "#F9A825"},
    {"name": "Poor",         "min": 201, "max": 300, "color": "#EF6C00"},
    {"name": "Very Poor",    "min": 301, "max": 400, "color": "#D32F2F"},
    {"name": "Severe",       "min": 401, "max": 500, "color": "#7B1FA2"},
]

MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "ANN",
        "Linear Regression",
        "Tuned XGBoost",
        "Tuned Random Forest",
        "Random Forest",
        "XGBoost",
        "KNN",
        "SVR",
        "Decision Tree",
    ],
    "MAE": [
        18.8214, 19.1595, 18.8763, 19.7752,
        19.9423, 20.2066, 20.5792, 24.1968, 29.4179,
    ],
    "RMSE": [
        28.3949, 28.7383, 28.7655, 30.1088,
        30.1295, 30.2184, 30.9606, 35.8222, 44.8834,
    ],
    "R² Score": [
        0.7662, 0.7605, 0.7600, 0.7371,
        0.7367, 0.7352, 0.7220, 0.6279, 0.4158,
    ],
})

PLOTLY_COLORS = [
    "#2E7D32", "#43A047", "#66BB6A", "#81C784", "#A5D6A7",
    "#1B5E20", "#388E3C", "#4CAF50", "#C8E6C9", "#E8F5E9",
]

@st.cache_resource(show_spinner=False)
def load_model():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout

    model = Sequential([
        Input(shape=(19,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="linear"),
    ])
    model.load_weights(MODEL_PATH)
    return model

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH)

def prepare_features(
    pollutant: str,
    year: int,
    month: int,
    day: int,
    aqi_lag_1: float,
    aqi_lag_7: float,
) -> pd.DataFrame:
    weekday = datetime(year, month, day).weekday()
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = {
        "No. Stations": 1,
        "year": year,
        "day": day,
        "month": month,
        "weekdays": weekday,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "AQI_lag_1": aqi_lag_1,
        "AQI_lag_week_ago": aqi_lag_7,
    }

    pollutant_cols = [c for c in FEATURE_COLUMNS if c.startswith("Prominent Pollutant_")]
    for col in pollutant_cols:
        col_pollutant = col.replace("Prominent Pollutant_", "")
        features[col] = 1 if pollutant == col_pollutant else 0

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)

def predict_aqi(features_df: pd.DataFrame) -> float:
    scaler = load_scaler()
    model = load_model()
    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled, verbose=0)
    return float(prediction[0][0])

def get_aqi_category(aqi_value: float) -> tuple[str, str]:
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_value <= cat["max"]:
            return cat["name"], cat["color"]
    if aqi_value > 500:
        return "Severe", "#7B1FA2"
    return "Good", "#388E3C"

def plotly_layout(title: str = "", height: int = 420, **kwargs) -> dict:
    layout = dict(
        title=dict(
            text=title,
            font=dict(size=15, color="#1E293B", family="Inter, sans-serif"),
            x=0.0,
            xanchor="left",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        height=height,
        margin=dict(l=48, r=24, t=56, b=48),
        xaxis=dict(
            gridcolor="#F1F5F9",
            gridwidth=0.5,
            linecolor="#E2E8F0",
            linewidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            gridwidth=0.5,
            linecolor="#E2E8F0",
            linewidth=1,
            zeroline=False,
        ),
        colorway=PLOTLY_COLORS,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#E2E8F0",
            font=dict(family="Inter, sans-serif", size=12, color="#1E293B"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    layout.update(kwargs)
    return layout

def inject_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)

def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "#2E7D32",
) -> str:
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-header">'
        f'    <span class="kpi-title">{title}</span>'
        f"  </div>"
        f'  <div class="kpi-value" style="color:{color};">{value}</div>'
        f'  <div class="kpi-subtitle">{subtitle}</div>'
        f"</div>"
    )

def render_page_header(title: str, description: str = ""):
    html = f'<div class="page-header"><h1>{title}</h1>'
    if description:
        html += f"<p>{description}</p>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_prediction_sidebar():
    with st.sidebar:
        st.markdown("### AQI Prediction")
        st.markdown("---")

        st.selectbox("City", ["Nashik"], key="sb_city")

        pollutant = st.selectbox(
            "Prominent Pollutant",
            POLLUTANT_CATEGORIES,
            index=POLLUTANT_CATEGORIES.index("PM10"),
            key="sb_pollutant",
        )

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input(
                "Year", min_value=2016, max_value=2035, value=2025, key="sb_year"
            )
        with c2:
            month = st.number_input(
                "Month", min_value=1, max_value=12, value=1, key="sb_month"
            )

        day = st.number_input(
            "Day", min_value=1, max_value=31, value=1, key="sb_day"
        )

        aqi_lag_1 = st.number_input(
            "AQI Lag 1 (Yesterday)",
            min_value=0.0,
            max_value=500.0,
            value=80.0,
            step=1.0,
            key="sb_lag1",
        )
        aqi_lag_7 = st.number_input(
            "AQI Lag 7 (Week Ago)",
            min_value=0.0,
            max_value=500.0,
            value=85.0,
            step=1.0,
            key="sb_lag7",
        )

        predict_btn = st.button(
            "Predict AQI",
            type="primary",
            use_container_width=True,
            key="sb_predict",
        )

        if predict_btn:
            with st.spinner("Running prediction..."):
                try:
                    features = prepare_features(
                        pollutant, year, month, day, aqi_lag_1, aqi_lag_7
                    )
                    predicted = predict_aqi(features)
                    category, color = get_aqi_category(predicted)

                    st.session_state["predicted_aqi"] = predicted
                    st.session_state["predicted_category"] = category
                    st.session_state["predicted_color"] = color
                    st.session_state["prediction_inputs"] = {
                        "pollutant": pollutant,
                        "year": year,
                        "month": month,
                        "day": day,
                        "aqi_lag_1": aqi_lag_1,
                        "aqi_lag_7": aqi_lag_7,
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        if "predicted_aqi" in st.session_state:
            st.markdown("---")
            aqi = st.session_state["predicted_aqi"]
            cat = st.session_state["predicted_category"]
            clr = st.session_state["predicted_color"]
            st.markdown(
                f'<div style="text-align:center;padding:14px;'
                f"background:{clr}12;border-left:4px solid {clr};"
                f'border-radius:8px;margin-top:4px;">'
                f'<div style="font-size:11px;font-weight:600;color:#64748B;'
                f'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;">'
                f"Predicted AQI</div>"
                f'<div style="font-size:28px;font-weight:700;color:{clr};">'
                f"{aqi:.1f}</div>"
                f'<div style="font-size:12px;color:{clr};font-weight:600;'
                f'margin-top:4px;">{cat}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

def setup_page(title: str = "Air Quality Analysis", layout: str = "wide"):
    st.set_page_config(
        page_title=title,
        page_icon="🌿",
        layout=layout,
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_prediction_sidebar()
