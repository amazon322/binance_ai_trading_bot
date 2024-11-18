import pandas as pd
import logging
import os
import joblib
from prophet import Prophet
import numpy as np
from scipy import stats

def train_prophet_model(
    data, model_save_path, model_name
):
    try:
        logging.info("Starting Prophet model training...")

        # Prepare the data
        df = pd.DataFrame(data, columns=['y'])
        df['ds'] = pd.date_range(start='2000-01-01', periods=len(data), freq='H')  # Adjust date range and frequency if needed

        # Remove anomalies or outliers using Z-score
        df['z_score'] = np.abs(stats.zscore(df['y']))
        df = df[df['z_score'] < 3]
        df.drop('z_score', axis=1, inplace=True)

        # Optionally, adjust the scale if required
        # Uncomment and adjust divisor if needed
        # df['y'] = df['y'] / 1e8  # Convert from Satoshi to BTC if necessary

        # Remove timezone information
        df['ds'] = df['ds'].dt.tz_localize(None)

        # Initialize the model with improved configuration
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='additive',
            changepoint_prior_scale=0.5  # Adjust as needed
        )
        # Add monthly seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Fit the model
        model.fit(df)

        # Save the model
        os.makedirs(model_save_path, exist_ok=True)
        model_path = os.path.join(model_save_path, model_name)
        joblib.dump(model, model_path)
        logging.info("Prophet model training completed and saved successfully.")
        return model

    except Exception as e:
        logging.exception("An error occurred during Prophet model training.")
        raise e