# Air Quality Analysis for Clean Air & Pollution Prevention

This is a Streamlit dashboard built for my B.E. Artificial Intelligence & Data Science project. It analyzes historical air quality data (AQI) from Nashik, Maharashtra, and uses a trained Artificial Neural Network (ANN) to predict the AQI.

🔗 **Live Application:** [View the Dashboard on Render](https://nashik-aqi-prediction.onrender.com)

## Project Structure

* `app.py`: Main entry point for the Streamlit dashboard.
* `utils.py`: Contains common functions for model predictions, preprocessing, and UI elements.
* `requirements.txt`: Python libraries needed to run this project.
* `app/`: Contains the pre-trained ANN model (`aqi_prediction_ann.keras`) and standard scaler (`scaler.pkl`).
* `dataset/`: Contains the raw and processed AQI dataset files.
* `notebooks/`: Jupyter notebooks with EDA, feature engineering, and model training code.
* `pages/`: Files for each section/page of the dashboard (Dashboard, Data Explorer, Visualizations, AQI Prediction, Model Performance, About Project).

## Setup & Run

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

Note: If you run into TensorFlow installation issues on Windows, you might need to enable Windows long path support.

## Model Performance Summary

The project evaluated multiple machine learning models on the Nashik AQI dataset. The Artificial Neural Network (ANN) performed the best and was selected for predictions in this dashboard.

* **ANN**: MAE = 18.82, RMSE = 28.39, R² = 0.766
* **Linear Regression**: MAE = 19.16, RMSE = 28.74, R² = 0.760
* **Tuned XGBoost**: MAE = 18.88, RMSE = 28.77, R² = 0.760
* **Tuned Random Forest**: MAE = 19.78, RMSE = 30.11, R² = 0.737
* **KNN**: MAE = 20.58, RMSE = 30.96, R² = 0.722
* **SVR**: MAE = 24.20, RMSE = 35.82, R² = 0.628
* **Decision Tree**: MAE = 29.42, RMSE = 44.88, R² = 0.416

## Future Work
* Connecting to live AQI data feeds for real-time predictions.
* Adding weather parameters like temperature, wind speed, and humidity to the model.
* Expanding predictions to other cities.
