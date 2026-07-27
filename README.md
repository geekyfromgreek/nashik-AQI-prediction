# Air Quality Analysis for Clean Air & Pollution Prevention

A Streamlit-based analytics dashboard built for my B.E. Artificial Intelligence & Data Science project. It analyzes historical air quality data (AQI) from Nashik, Maharashtra, and uses a trained Artificial Neural Network (ANN) to predict the AQI based on date, pollutant type, and recent AQI history.

🔗 **Live App:** [nashik-aqi-prediction.onrender.com](https://nashik-aqi-prediction.onrender.com)

## Features

- **Dashboard** — KPI cards showing overall AQI stats, yearly trend line chart, and air quality category breakdown.
- **Data Explorer** — Browse and filter the raw and processed datasets interactively.
- **Visualizations** — Monthly AQI trend, AQI distribution histogram, yearly trend, pollutant frequency, actual vs predicted scatter plot, training loss curve, model comparison bar chart, and feature correlation heatmap.
- **AQI Prediction** — Enter pollutant, date, and recent AQI values in the sidebar to get a real-time prediction from the trained ANN model.
- **Model Performance** — Side-by-side comparison of all trained models (ANN, Linear Regression, XGBoost, Random Forest, KNN, SVR, Decision Tree) with MAE, RMSE, and R² scores.
- **About Project** — Background information on the project, dataset, and methodology.

## Project Structure

```
├── app.py                  # Streamlit entry point
├── utils.py                # Shared helpers (model loading, preprocessing, UI components)
├── requirements.txt        # Python dependencies
├── app/
│   ├── aqi_prediction_ann.keras   # Trained ANN model (weights + config)
│   └── scaler.pkl                 # Fitted StandardScaler
├── dataset/
│   ├── Nashik_AQIBulletins.csv    # Raw AQI data
│   └── processed_aqi_data.csv     # Feature-engineered data
├── notebooks/
│   ├── EDA-nashik-aqi-prediction.ipynb
│   ├── feature-extraction-nashik-aqi.ipynb
│   └── model-training-evaluation-nashik-aqi.ipynb
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Data_Explorer.py
│   ├── 3_Visualizations.py
│   ├── 4_AQI_Prediction.py
│   ├── 5_Model_Performance.py
│   └── 6_About_Project.py
└── assets/
    └── style.css
```

## Setup & Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **ANN** | **18.82** | **28.39** | **0.766** |
| Linear Regression | 19.16 | 28.74 | 0.760 |
| Tuned XGBoost | 18.88 | 28.77 | 0.760 |
| Tuned Random Forest | 19.78 | 30.11 | 0.737 |
| Random Forest | 19.94 | 30.13 | 0.737 |
| XGBoost | 20.21 | 30.22 | 0.735 |
| KNN | 20.58 | 30.96 | 0.722 |
| SVR | 24.20 | 35.82 | 0.628 |
| Decision Tree | 29.42 | 44.88 | 0.416 |

The ANN achieved the lowest MAE and RMSE and the highest R², so it was selected for the prediction feature.

## Tech Stack

- Python 3.10+
- Streamlit
- TensorFlow / Keras
- Plotly
- Pandas, NumPy, Scikit-learn, XGBoost

## Future Work

- Connect to live AQI data feeds for real-time monitoring.
- Add weather parameters (temperature, wind speed, humidity) as model features.
- Expand predictions to other cities in Maharashtra.
