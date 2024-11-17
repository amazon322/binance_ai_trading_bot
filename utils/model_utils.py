from config import Config
import os
import joblib
import torch
import numpy as np
import pandas as pd
from flask import render_template
import torch.nn as nn
def parse_prediction_form(request):
    """
    Extracts and validates form data from the prediction request.

    Parameters:
    - request (flask.Request): The Flask request object containing form data.

    Returns:
    - model_type (str): The type of the model selected by the user.
    - coin (str): The coin pair entered by the user.
    - forecast_horizon (int): The number of future time steps to predict.
    - include_sentiment (bool): Whether to include sentiment analysis.

    Raises:
    - ValueError: If required form data is missing or invalid.
    """
    # Extract form data
    model_file = request.form.get('model_file')  # Expected format: 'modeltype_coinpair'
    include_sentiment = request.form.get('include_sentiment') == 'on'
    coin = request.form.get('coin')
    forecast_horizon_str = request.form.get('forecast_horizon', '24')

    # Validate form data
    if not model_file:
        raise ValueError("Please select a model.")

    if not coin:
        raise ValueError("Please enter a coin pair.")

    try:
        forecast_horizon = int(forecast_horizon_str)
    except ValueError:
        raise ValueError("Forecast horizon must be an integer.")

    if forecast_horizon <= 0:
        raise ValueError("Forecast horizon must be a positive integer.")

    # Extract model_type and coin_from_file using rsplit
    model_type_from_file, coin_from_file_with_ext = model_file.rsplit('_', 1)
    coin_from_file = coin_from_file_with_ext.split('.')[0]

    if coin_from_file != coin:
        raise ValueError(f"The selected model is for {coin_from_file}, but you entered {coin}. "
                         f"Please ensure they match.")

    model_type = model_type_from_file.lower()

    return model_type, coin, forecast_horizon, include_sentiment

def load_model_and_scaler(model_type, coin):
    """
    Loads the trained model and scaler for the specified model type and coin.

    Parameters:
    - model_type (str): The type of the model (e.g., 'lstm', 'arima').
    - coin (str): The coin pair for which the model was trained.

    Returns:
    - model_info (dict or object): Model information or the model itself.
    - scaler (object or None): The scaler used during training (if applicable).

    Raises:
    - FileNotFoundError: If the model or scaler files are not found.
    """
    model_save_path = Config.MODEL_SAVE_PATH

    if model_type in ['arima', 'garch']:
        model_file_path = os.path.join(model_save_path, f'{model_type}_{coin}.pkl')
    else:
        model_file_path = os.path.join(model_save_path, f'{model_type}_{coin}.pth')

    scaler_file = os.path.join(model_save_path, f'scaler_{model_type}_{coin}.pkl')

    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"No trained model found for {model_type.upper()} on {coin}. "
                                f"Please train the model first.")

    # Load the model
    if model_type in ['arima', 'garch']:
        # For ARIMA and GARCH models saved with joblib
        model = joblib.load(model_file_path)
        model_info = model  # For consistency
    else:
        # For PyTorch models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_info = torch.load(model_file_path, map_location=device,weights_only=True)

        if not os.path.exists(scaler_file):
            raise FileNotFoundError("Scaler file not found. Please retrain the model.")

    # Load the scaler if applicable
    if model_type in ['arima', 'garch']:
        scaler = None  # Scaler is not used for ARIMA and GARCH models
    else:
        scaler = joblib.load(scaler_file)

    return model_info, scaler

def prepare_prediction_data(df, scaler, model_info, data_processor, sequence_length):
    """
    Prepares the input data for prediction by applying preprocessing steps.

    Parameters:
    - df (pd.DataFrame): DataFrame containing historical data.
    - scaler (object): The scaler used during training.
    - model_info (dict): Dictionary containing model parameters and state dict.
    - data_processor (DataProcessor): Instance of the DataProcessor class.
    - sequence_length (int): The sequence length used during training.

    Returns:
    - input_sequence (np.ndarray): The prepared input sequence for prediction.

    Raises:
    - ValueError: If there is insufficient data for making a prediction.
    """
    # Extract the features used during training
    feature_columns = model_info.get('feature_columns', ['close'])
    data_features = df[feature_columns].values

    # Scale data using the loaded scaler
    scaled_data = scaler.transform(data_features)

    # Ensure sufficient data
    if len(scaled_data) < sequence_length:
        raise ValueError("Insufficient data for making a prediction.")

    # Prepare input sequence
    input_sequence = scaled_data[-sequence_length:]
    input_sequence = np.expand_dims(input_sequence, axis=0)
    return input_sequence

def make_predictions(model_type, model_info, scaler, data_series, input_sequence, forecast_horizon, device):
    """
    Makes predictions using the specified model type.

    Parameters:
    - model_type (str): The type of the model ('lstm', 'cnn_lstm', 'transformer').
    - model_info (dict): Model information or the model itself.
    - scaler (object): The scaler used during training.
    - data_series (np.ndarray): Array of historical price data.
    - input_sequence (np.ndarray): The prepared input sequence for prediction.
    - forecast_horizon (int): Number of future time steps to predict.
    - device (torch.device): Device to perform computation on.

    Returns:
    - predicted_prices (np.ndarray): Array of predicted prices.
    """
    if model_type == 'lstm':
        predicted_prices = predict_with_lstm(
            model_info, scaler, input_sequence, forecast_horizon, device
        )
    elif model_type == 'cnn_lstm':
        predicted_prices = predict_with_cnn_lstm(
            model_info, scaler, input_sequence, forecast_horizon, device
        )
    elif model_type == 'transformer':
        predicted_prices = predict_with_transformer(
            model_info, scaler, input_sequence, forecast_horizon, device
        )
    else:
        raise ValueError(f"Prediction for model '{model_type}' is not implemented.")
    return predicted_prices

def adjust_for_sentiment(predicted_prices, coin):
    """
    Performs sentiment analysis and adjusts predictions accordingly.

    Parameters:
    - predicted_prices (np.ndarray): Array of predicted prices.
    - coin (str): The coin pair for which sentiment analysis is performed.

    Returns:
    - adjusted_prices (np.ndarray): The sentiment-adjusted predicted prices.
    - sentiment_label (str): The sentiment label ('Bullish', 'Bearish', or 'Neutral').
    - articles (list): List of news articles related to the coin.
    - average_sentiment (float): The average sentiment score.
    """
    from data.sentiment_analysis import get_news_articles, analyze_sentiment

    articles = get_news_articles(coin)
    sentiments = analyze_sentiment(articles)
    if sentiments:
        average_sentiment = sum(sentiments) / len(sentiments)
        # Apply sentiment adjustment
        sentiment_factor = 1 + np.clip(average_sentiment, -0.2, 0.2) * 0.05  # Adjust scaling as needed
        adjusted_prices = predicted_prices * sentiment_factor
        sentiment_label = (
            'Bullish' if average_sentiment > 0 else 'Bearish' if average_sentiment < 0 else 'Neutral'
        )
    else:
        average_sentiment = 0.0
        sentiment_label = 'Neutral'
        adjusted_prices = predicted_prices  # No adjustment

    return adjusted_prices, sentiment_label, articles, average_sentiment
# utils/model_utils.py

def predict_with_lstm(model_info, scaler, input_sequence, forecast_horizon, device):
    """
    Predict future prices using an LSTM model.

    Parameters:
    - model_info (dict): Model parameters and state dict.
    - scaler (object): Scaler used during training.
    - input_sequence (np.ndarray): Prepared input sequence for prediction.
    - forecast_horizon (int): Number of future time steps to predict.
    - device (torch.device): Device for computation.

    Returns:
    - predicted_prices (np.ndarray): Predicted prices.
    """
    from models.lstm import LSTMModel

    # Initialize the model
    input_size = model_info.get('input_size', 1)
    hidden_size = model_info.get('hidden_size', 50)
    num_layers = model_info.get('num_layers', 2)
    dropout = model_info.get('dropout', 0.1)
    state_dict = model_info['state_dict']
    sequence_length = model_info.get('sequence_length', 60)  # Default to 60 if not specified

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Predict multiple future steps
    predicted_prices = make_multi_step_predictions(
        model, input_sequence, scaler, forecast_horizon, device
    )
    return predicted_prices

def generate_prediction_plot(predicted_prices, coin):
    """
    Generates a plot of the predicted prices and returns the plot URL.

    Parameters:
    - predicted_prices (np.ndarray): Array of predicted prices.
    - coin (str): The coin pair for which the prediction is made.

    Returns:
    - plot_url (str): The base64 encoded URL of the plot image.
    """
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_prices, marker='o')
    plt.title(f'Predicted Prices for {coin}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def render_prediction_result(template, **kwargs):
    """
    Renders the result template with provided data.

    Parameters:
    - template (str): The name of the template to render.
    - **kwargs: Additional keyword arguments to pass to the template.

    Returns:
    - response (flask.Response): The rendered template response.
    """
    return render_template(template, **kwargs)

def prepare_input_sequence(data_series, scaler, sequence_length):
    # Scale data
    scaled_data = scaler.transform(data_series)
    
    # Ensure sufficient data
    if len(scaled_data) < sequence_length:
        raise ValueError("Insufficient data for making a prediction.")
    
    # Prepare input sequence
    input_sequence = scaled_data[-sequence_length:]
    input_sequence = np.expand_dims(input_sequence, axis=0)
    return input_sequence

def make_multi_step_predictions(model, input_sequence, scaler, forecast_horizon, device):
    """
    Makes multi-step predictions using the provided model.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to use for predictions.
    - input_sequence (np.ndarray): The prepared input sequence for prediction.
    - scaler (object): The scaler used during training.
    - forecast_horizon (int): The number of future time steps to predict.
    - device (torch.device): Device to perform computation on.

    Returns:
    - predicted_prices (np.ndarray): Array of predicted prices.
    """
    predictions = []
    input_seq = input_sequence.copy()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            input_tensor = torch.from_numpy(input_seq).float().to(device)
            pred = model(input_tensor)
            pred_value = pred.cpu().numpy()[0][0]
            predictions.append(pred_value)
            # Update input sequence
            input_seq = np.append(input_seq[:, 1:, :], [[[pred_value]]], axis=1)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predicted_prices = predictions.flatten()
    return predicted_prices



def predict_with_cnn_lstm(model_info, scaler, input_sequence, forecast_horizon, device):
    from models.hybrid_cnn_lstm import CNNLSTMModel

    # Extract model parameters from model_info
    input_size = model_info['input_size']
    hidden_size = model_info['hidden_size']
    num_layers = model_info['num_layers']
    sequence_length = model_info['sequence_length']
    dropout = model_info.get('dropout', 0.0)
    state_dict = model_info['state_dict']

    # Ensure dropout is a float
    if isinstance(dropout, nn.Dropout):
        dropout = dropout.p
        
    # Initialize the model
    model = CNNLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prediction loop
    predictions = []
    input_seq = input_sequence.copy()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            input_tensor = torch.from_numpy(input_seq).float().to(device)
            pred = model(input_tensor)
            pred_value = pred.cpu().numpy()[0][0]
            predictions.append(pred_value)
            # Update input sequence
            next_input = np.append(input_seq[:, 1:, :], [[[pred_value]]], axis=1)
            input_seq = next_input

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predicted_prices = predictions.flatten()
    return predicted_prices


        ### Other Functions ###

def predict_with_transformer(model_info, scaler, input_sequence, forecast_horizon, device):
    from models.transformer import TimeSeriesTransformer
    import torch
    import numpy as np

    # Load model parameters
    input_size = model_info.get('input_size')
    sequence_length = model_info.get('sequence_length')
    state_dict = model_info['state_dict']
    model_params = model_info.get('model_params')

    model = TimeSeriesTransformer(
        input_size=input_size,
        num_encoder_layers=model_params.get('num_encoder_layers', 3),
        num_decoder_layers=model_params.get('num_decoder_layers', 3),
        dim_model=model_params.get('dim_model', 512),
        num_heads=model_params.get('num_heads', 8),
        dim_feedforward=model_params.get('dim_feedforward', 2048),
        dropout=model_params.get('dropout', 0.1),
    ).to(device)

    model.load_state_dict(model_info['state_dict'])
    model.eval()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Prepare input sequence
    input_seq = input_sequence.copy()
    input_seq = torch.Tensor(input_seq).float().to(device).permute(1, 0, 2)  # (seq_len, batch_size, features)

    predicted_prices = []

    with torch.no_grad():
        for _ in range(forecast_horizon):
            tgt_input = input_seq[-1:, :, :]  # Last time step
            output = model(input_seq, tgt_input)
            prediction = output.cpu().numpy()
            predicted_prices.append(prediction[-1, 0, 0])  # Assuming output shape is (seq_len, batch_size, 1)
            # Update input sequence
            next_input = output[-1:, :, :]
            input_seq = torch.cat((input_seq, next_input), dim=0)

    # Inverse transform predictions
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    predicted_prices = predicted_prices.flatten()

    return predicted_prices

# utils/model_utils.py

def predict_with_prophet(model, data_series, forecast_horizon):
    import pandas as pd

    # Determine the last date in the data
    last_date = data_series.index[-1]

    # Prepare the future dataframe
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq='H'
    )
    future = pd.DataFrame({'ds': future_dates})

    # Make prediction
    forecast = model.predict(future)
    predicted_prices = forecast['yhat'].values
    return predicted_prices


# utils/model_utils.py

def generate_prediction_plot(predicted_prices, coin):
    import matplotlib
    matplotlib.use('Agg')  # Use a non-interactive backend
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_prices, marker='o')
    plt.title(f'Predicted Prices for {coin}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    image_data = img.getvalue()
    
    if len(image_data) == 0:
        print("Error: Image data is empty.")
    
    plot_url = base64.b64encode(image_data).decode()
    return plot_url
