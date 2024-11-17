# models/prophet_model.py
import pandas as pd
import logging
import os
import joblib
from prophet import Prophet

def train_prophet_model(
    data, model_save_path, model_name
):
    try:
        logging.info("Starting Prophet model training...")
        # Prepare the data
        df = pd.DataFrame(data, columns=['y'])
        df['ds'] = pd.date_range(start='2000-01-01', periods=len(data), freq='H')  # Adjust the date range and frequency

        # Initialize the model
        model = Prophet()
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