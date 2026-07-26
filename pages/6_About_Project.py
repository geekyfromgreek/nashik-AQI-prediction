import streamlit as st
from utils import (
    inject_css,
    render_page_header,
    render_prediction_sidebar,
)

st.set_page_config(
    page_title="About — AQI Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_prediction_sidebar()

render_page_header(
    "About This Project",
    "Air Quality Analysis for Clean Air and Pollution Prevention — B.E. Artificial Intelligence & Data Science",
)

st.markdown("## Project Overview")
st.markdown(
    '<div class="section-card">'
    "<p style='color:#334155;font-size:14px;line-height:1.8;'>"
    "This project analyses air quality data from Nashik, Maharashtra "
    "to understand pollution patterns, identify dominant pollutants, and predict the "
    "Air Quality Index (AQI) using machine learning and deep learning models. "
    "The goal is to support data-driven decision making for clean air initiatives "
    "and pollution prevention strategies."
    "</p></div>",
    unsafe_allow_html=True,
)

st.markdown("## Objectives")
objectives = [
    "Analyse historical AQI data for Nashik city to uncover trends and seasonal patterns.",
    "Identify the most prominent pollutants affecting air quality.",
    "Engineer meaningful features (temporal, cyclic, lag) to improve model accuracy.",
    "Train and compare multiple regression models for AQI prediction.",
    "Select the best-performing model and deploy it in an interactive dashboard.",
    "Provide actionable insights for pollution prevention and public health awareness.",
]
st.markdown(
    '<div class="section-card">'
    "<ul style='color:#334155;font-size:14px;line-height:2;padding-left:18px;'>"
    + "".join(f"<li>{obj}</li>" for obj in objectives)
    + "</ul></div>",
    unsafe_allow_html=True,
)

st.markdown("## Dataset Description")
st.markdown(
    '<div class="section-card">'
    '<table style="width:100%;font-size:13px;color:#334155;">'
    '<tr style="border-bottom:1px solid #F1F5F9;">'
    '<td style="padding:10px 4px;color:#64748B;font-weight:500;width:200px;">Source</td>'
    '<td style="padding:10px 4px;font-weight:500;">CPCB AQI Bulletins — Nashik</td></tr>'
    '<tr style="border-bottom:1px solid #F1F5F9;">'
    '<td style="padding:10px 4px;color:#64748B;font-weight:500;">Records</td>'
    '<td style="padding:10px 4px;font-weight:500;">2,412 daily observations</td></tr>'
    '<tr style="border-bottom:1px solid #F1F5F9;">'
    '<td style="padding:10px 4px;color:#64748B;font-weight:500;">Period</td>'
    '<td style="padding:10px 4px;font-weight:500;">2016 – 2024</td></tr>'
    '<tr style="border-bottom:1px solid #F1F5F9;">'
    '<td style="padding:10px 4px;color:#64748B;font-weight:500;">Columns</td>'
    '<td style="padding:10px 4px;font-weight:500;">date, City, No. Stations, Air Quality, Index Value, Prominent Pollutant</td></tr>'
    '<tr>'
    '<td style="padding:10px 4px;color:#64748B;font-weight:500;">Target Variable</td>'
    '<td style="padding:10px 4px;font-weight:500;">Index Value (AQI)</td></tr>'
    "</table></div>",
    unsafe_allow_html=True,
)

st.markdown("## Machine Learning Workflow")

steps = [
    ("1. Data Collection", "Raw CPCB AQI bulletins collected for Nashik city."),
    ("2. Exploratory Data Analysis", "Distribution analysis, trend plots, outlier detection, and category breakdowns."),
    ("3. Feature Engineering", "Temporal features (year, month, day, weekday), cyclic encoding (month_sin, month_cos), lag features (1-day, 7-day), one-hot encoding of prominent pollutants."),
    ("4. Preprocessing", "Missing value handling, datetime parsing, feature selection, train-test split (80/20), StandardScaler normalisation."),
    ("5. Model Training", "9 regression models trained including Linear Regression, Random Forest, XGBoost, and ANN."),
    ("6. Evaluation", "Models compared on MAE, RMSE, and R² Score. ANN achieved the best performance."),
    ("7. Deployment", "ANN model saved as .keras file with StandardScaler. Deployed via Streamlit dashboard."),
]

html = '<div class="info-grid">'
for title, desc in steps:
    html += (
        '<div class="info-item">'
        f"<h4>{title}</h4>"
        f"<p>{desc}</p>"
        "</div>"
    )
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

st.markdown("## Models Used")

models_html = '<div class="section-card">'
models_html += '<table style="width:100%;font-size:13px;color:#334155;">'
models_info = [
    ("Linear Regression", "Baseline linear model with StandardScaler"),
    ("Decision Tree Regressor", "Non-linear tree-based model"),
    ("Random Forest Regressor", "Ensemble of decision trees (default params)"),
    ("Tuned Random Forest", "Hyperparameter-tuned (n_estimators=200, max_depth=10)"),
    ("K-Nearest Neighbors (KNN)", "Instance-based learning with StandardScaler"),
    ("Support Vector Regressor (SVR)", "Kernel-based regression with StandardScaler"),
    ("XGBoost Regressor", "Gradient boosted trees (default params)"),
    ("Tuned XGBoost", "Tuned (n_estimators=1000, lr=0.01, max_depth=3)"),
    ("Artificial Neural Network", "2 hidden layers (128→64), ReLU, Dropout 0.2, Adam optimiser"),
]
for name, desc in models_info:
    bg = "background:#E8F5E9;" if name == "Artificial Neural Network" else ""
    models_html += (
        f'<tr style="border-bottom:1px solid #F1F5F9;{bg}">'
        f'<td style="padding:10px 8px;font-weight:600;width:240px;">{name}</td>'
        f'<td style="padding:10px 8px;color:#64748B;">{desc}</td></tr>'
    )
models_html += "</table></div>"
st.markdown(models_html, unsafe_allow_html=True)

st.markdown("## Feature Engineering")
st.markdown(
    '<div class="section-card">'
    "<ul style='color:#334155;font-size:14px;line-height:2;padding-left:18px;'>"
    "<li><strong>Temporal features:</strong> year, month, day, weekday extracted from the date column.</li>"
    "<li><strong>Cyclic encoding:</strong> month_sin = sin(2π × month / 12), month_cos = cos(2π × month / 12) to capture seasonal periodicity.</li>"
    "<li><strong>Lag features:</strong> AQI_lag_1 (1-day lag), AQI_lag_week_ago (7-day lag) to capture temporal auto-correlation.</li>"
    "<li><strong>One-hot encoding:</strong> Prominent Pollutant encoded using pd.get_dummies(drop_first=True). CO is the reference category. 10 binary columns created.</li>"
    "<li><strong>City encoding:</strong> Only Nashik present; column dropped after get_dummies(drop_first=True).</li>"
    "</ul></div>",
    unsafe_allow_html=True,
)

st.markdown("## ANN Architecture")
st.markdown(
    '<div class="section-card">'
    "<p style='color:#334155;font-size:14px;line-height:1.8;'>"
    "The Artificial Neural Network uses a Sequential architecture with two hidden layers. "
    "Dropout regularisation (rate 0.2) is applied after each hidden layer to prevent overfitting. "
    "The network was trained for 100 epochs with a batch size of 32 and 20% validation split.</p>"
    '<pre style="background:#F7F9F7;padding:16px;border-radius:8px;font-size:13px;'
    'color:#1E293B;line-height:1.7;margin-top:12px;border:1px solid #E2E8F0;">'
    "Input (19 features)\n"
    "  │\n"
    "  ├─ Dense(128, activation='relu')\n"
    "  ├─ Dropout(0.2)\n"
    "  │\n"
    "  ├─ Dense(64, activation='relu')\n"
    "  ├─ Dropout(0.2)\n"
    "  │\n"
    "  └─ Dense(1)  →  AQI Prediction\n"
    "\n"
    "Optimizer: Adam\n"
    "Loss: Mean Squared Error (MSE)\n"
    "Metrics: MSE, MAE"
    "</pre></div>",
    unsafe_allow_html=True,
)

st.markdown("## Conclusion")
st.markdown(
    '<div class="section-card">'
    "<p style='color:#334155;font-size:14px;line-height:1.8;'>"
    "The ANN achieved the best overall performance among all evaluated models, "
    "with an MAE of 18.82, RMSE of 28.39, and R² Score of 0.7662. "
    "Engineered features, particularly lag features and cyclic month encoding, "
    "significantly improved prediction accuracy across all models."
    "</p></div>",
    unsafe_allow_html=True,
)

st.markdown("## Future Scope")
future_items = [
    "Incorporate real-time AQI data feeds for live monitoring and prediction.",
    "Extend coverage to multiple cities and stations across Maharashtra.",
    "Integrate weather parameters (temperature, humidity, wind) as additional features.",
    "Experiment with advanced architectures (LSTM, Transformer) for time-series forecasting.",
    "Implement early warning alerts for hazardous air quality conditions.",
    "Deploy as a cloud-hosted application with public access.",
    "Add pollutant-specific prediction models for granular analysis.",
]
st.markdown(
    '<div class="section-card">'
    "<ul style='color:#334155;font-size:14px;line-height:2;padding-left:18px;'>"
    + "".join(f"<li>{item}</li>" for item in future_items)
    + "</ul></div>",
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:12px;'>"
    "B.E. Artificial Intelligence &amp; Data Science — Air Quality Analysis Project"
    "</p>",
    unsafe_allow_html=True,
)
