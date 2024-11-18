from flask import Blueprint, render_template, request, redirect, url_for, flash
from config import Config
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from utils.model_utils import (
    parse_prediction_form,
    load_model_and_scaler,
    prepare_prediction_data,
    make_predictions,
    predict_with_prophet,
    predict_with_cnn_lstm,
    adjust_for_sentiment,
    generate_prediction_plot,
    render_prediction_result
)
# Import torch and other necessary libraries
import torch
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta, timezone
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from models.hybrid_cnn_lstm import train_cnn_lstm_model
from models.lstm import train_lstm_model
# Add new imports
from models.transformer import train_transformer_model
from models.prophet_model import train_prophet_model
import joblib
import pytz
# Configure logging
logging.basicConfig(level=logging.INFO)

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/')
def index():
    return render_template('index.html')

@main_routes.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# app_routes/main_routes.py

@main_routes.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        # Handle form submission for training a model
        model_type = request.form.get('model_type')
        coin = request.form.get('coin')
        data_period = int(request.form.get('data_period'))
        epochs = int(request.form.get('epochs'))
        batch_size = int(request.form.get('batch_size'))
        sequence_length = int(request.form.get('sequence_length'))

        # Extract hyperparameters
        hidden_size = int(request.form.get('hidden_size'))
        num_layers = int(request.form.get('num_layers'))
        dropout = float(request.form.get('dropout'))
        learning_rate = float(request.form.get('learning_rate'))
        weight_decay = float(request.form.get('weight_decay'))

        # Fetch historical data
        data_loader = DataLoader()
        data_processor = DataProcessor()

        # Define the start and end dates based on data_period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=data_period)
        start_str = start_date.strftime('%d %b %Y %H:%M:%S')
        end_str = end_date.strftime('%d %b %Y %H:%M:%S')

        df = data_loader.get_historical_data(
            symbol=coin,
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str=start_str,
            end_str=end_str
        )

        if df is not None and not df.empty:
            # Preprocess data
            data_series = df['close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_series)

            # Prepare sequences for models that require them
            if model_type.lower() in ['lstm', 'cnn_lstm', 'transformer']:
                X, y = data_processor.create_sequences(scaled_data, sequence_length)
                # Split data into training and validation sets
                X_train, X_val, y_train, y_val = data_processor.split_data(X, y, train_size=0.8)
                input_size = X_train.shape[2]  # Number of features

            # Train the model based on the selected model type
            if model_type.lower() == 'lstm':
                train_lstm_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    model_save_path=Config.MODEL_SAVE_PATH,
                    model_name=f'{model_type}_{coin}.pth',
                    sequence_length=sequence_length,
                    scaler=scaler,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    weight_decay=weight_decay
                )
            elif model_type.lower() == 'cnn_lstm':
                from models.hybrid_cnn_lstm import train_cnn_lstm_model
                train_cnn_lstm_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=0.001,
                    model_save_path=Config.MODEL_SAVE_PATH,
                    model_name=f'{model_type}_{coin}.pth',
                    scaler=scaler,
                    sequence_length=sequence_length
                )

            elif model_type.lower() == 'transformer':
                input_size = X_train.shape[2]  # Number of features
                train_transformer_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    model_save_path=Config.MODEL_SAVE_PATH,
                    model_name=f'{model_type}_{coin}.pth',
                    scaler=scaler
                    # Removed 'sequence_length'
                )



            elif model_type.lower() == 'prophet':
                # Prophet model training
                from models.prophet_model import train_prophet_model
                data_series = df[['close']].copy()
                if 'timestamp' in df.columns:
                    data_series['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    data_series['ds'] = df.index  # Use index if timestamp is not available
                    data_series['ds'] = pd.to_datetime(data_series['ds'])
                data_series.rename(columns={'close': 'y'}, inplace=True)
                train_prophet_model(
                    data=data_series,
                    model_save_path=Config.MODEL_SAVE_PATH,
                    model_name=f'{model_type}_{coin}.pkl'
                )

            else:
                flash(f"Model type '{model_type}' is not supported.")
                return redirect(url_for('main_routes.train_model'))

            flash(f"Training completed for {model_type.upper()} model on {coin}.")
            return redirect(url_for('main_routes.dashboard'))
        else:
            flash("No data available for training.")
            return redirect(url_for('main_routes.train_model'))
    else:
        return render_template('train_model.html')

            ### Models for Prediction ####


@main_routes.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        model_file = request.form.get('model_file')  # Selected model from the form
        include_sentiment = request.form.get('include_sentiment') == 'on'
        forecast_horizon = int(request.form.get('forecast_horizon', 1))  # Get forecast horizon from user

        if not model_file:
            flash("Please select a model.")
            return redirect(url_for('main_routes.prediction'))

        model_file = os.path.splitext(model_file)[0]  # Remove file extension
        model_type, coin = model_file.rsplit('_', 1)
        model_file_path = os.path.join(Config.MODEL_SAVE_PATH, f'{model_type}_{coin}')
        # For Prophet models, the extension is '.pkl'; for others, '.pth'
        if model_type.lower() == 'prophet':
            model_file_path += '.pkl'
        else:
            model_file_path += '.pth'
        scaler_file = os.path.join(Config.MODEL_SAVE_PATH, f'scaler_{model_type}_{coin}.pkl')

        if not os.path.exists(model_file_path):
            flash(f"No trained model found for {model_type.upper()} on {coin}. Please train the model first.")
            return redirect(url_for('main_routes.train_model'))

        # Fetch recent historical data for prediction
        data_loader = DataLoader()
        data_processor = DataProcessor()

        # Get recent data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)  # Adjust as needed
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

        df = data_loader.get_historical_data(
            symbol=coin,
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str=start_str,
            end_str=end_str
        )

        if df is not None and not df.empty:
            data_series = df['close'].values.reshape(-1, 1)

            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                if model_type.lower() == 'lstm':
                    # Load scaler
                    scaler = joblib.load(scaler_file)
                    # Load model_info
                    model_info = torch.load(model_file_path, map_location=device)
                    input_size = model_info['input_size']
                    sequence_length = model_info['sequence_length']
                    # Prepare input sequence
                    scaled_data = scaler.transform(data_series)
                    input_sequence = scaled_data[-sequence_length:]
                    input_sequence = np.expand_dims(input_sequence, axis=0)
                    # Make predictions
                    predicted_prices = make_predictions(
                        model_type='lstm',
                        model_info=model_info,
                        scaler=scaler,
                        data_series=data_series,
                        input_sequence=input_sequence,
                        forecast_horizon=forecast_horizon,
                        device=device
                    )
                elif model_type.lower() == 'cnn_lstm':
                    # Load scaler
                    scaler = joblib.load(scaler_file)

                    # Load model info
                    model_info = torch.load(model_file_path, map_location=device)
                    input_size = model_info['input_size']
                    sequence_length = model_info['sequence_length']

                    # Prepare input sequence
                    scaled_data = scaler.transform(data_series)
                    input_sequence = scaled_data[-sequence_length:]
                    input_sequence = np.expand_dims(input_sequence, axis=0)

                    # Make predictions using make_predictions function
                    predicted_prices = make_predictions(
                        model_type='cnn_lstm',
                        model_info=model_info,
                        scaler=scaler,
                        data_series=data_series,
                        input_sequence=input_sequence,
                        forecast_horizon=forecast_horizon,
                        device=device
                    )


                elif model_type.lower() == 'transformer':
                    from models.transformer import TimeSeriesTransformer  # Ensure correct import

                    # Load the scaler
                    scaler_file = os.path.join(Config.MODEL_SAVE_PATH, f'scaler_{model_type}_{coin}.pkl')
                    if not os.path.exists(scaler_file):
                        flash("Scaler file not found. Please retrain the model.")
                        return redirect(url_for('main_routes.train_model'))
                    scaler = joblib.load(scaler_file)

                    # Load model info
                    model_info = torch.load(model_file_path, map_location=device)

                    # Scale data
                    scaled_data = scaler.transform(data_series)

                    # Prepare input sequence
                    sequence_length = model_info['sequence_length']
                    if len(scaled_data) < sequence_length:
                        flash("Insufficient data for making a prediction.")
                        return redirect(url_for('main_routes.prediction'))

                    input_sequence = scaled_data[-sequence_length:]
                    input_sequence = np.expand_dims(input_sequence, axis=0)


                    # Make predictions
                    predicted_prices = make_predictions(
                        model_type=model_type.lower(),
                        model_info=model_info,
                        scaler=scaler,
                        data_series=scaled_data,
                        input_sequence=input_sequence,
                        forecast_horizon=forecast_horizon,
                        device=device,
                        coin=coin
                    )
                elif model_type.lower() == 'prophet':
                    import numpy as np
                    from scipy import stats

                    # Load the Prophet model
                    model = joblib.load(model_file_path)

                    # Prepare data_series
                    data_series = df[['close']].copy()

                    # Handle the datetime information
                    if 'timestamp' in df.columns:
                        data_series['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
                    else:
                        data_series['ds'] = pd.to_datetime(df.index)

                    # Remove timezone information
                    data_series['ds'] = data_series['ds'].dt.tz_localize(None)

                    # Rename 'close' to 'y'
                    data_series.rename(columns={'close': 'y'}, inplace=True)

                    # Remove anomalies or outliers using Z-score
                    data_series['z_score'] = np.abs(stats.zscore(data_series['y']))
                    data_series = data_series[data_series['z_score'] < 3]
                    data_series.drop('z_score', axis=1, inplace=True)

                    # Optionally, adjust the scale if required (e.g., if prices are in Satoshi)
                    # Uncomment and adjust divisor if needed
                    # data_series['y'] = data_series['y'] / 1e8  # Convert from Satoshi to BTC if necessary

                    # Reorder columns
                    data_series = data_series[['ds', 'y']]

                    # Make predictions
                    predicted_prices = predict_with_prophet(
                        model=model,
                        data_series=data_series,
                        forecast_horizon=forecast_horizon
                    )


                else:
                    flash(f"Prediction for model '{model_type}' is not implemented.")
                    return redirect(url_for('main_routes.prediction'))

                # Adjust predictions based on sentiment if applicable
                if include_sentiment:
                    # Fetch and analyze sentiment data
                    from data.sentiment_analysis import get_news_articles, analyze_sentiment

                    articles = get_news_articles(coin)
                    sentiments = analyze_sentiment(articles)
                    sentiment_label = 'Neutral'
                    if sentiments:
                        average_sentiment = sum(sentiments) / len(sentiments)
                        if average_sentiment > 0.1:
                            sentiment_label = 'Positive'
                        elif average_sentiment < -0.1:
                            sentiment_label = 'Negative'
                    else:
                        average_sentiment = 0.0
                    # Adjust predictions (optional)
                    # For simplicity, we can just pass the sentiment label to the template
                else:
                    articles = []
                    sentiment_label = 'Not Included'

                # Generate the plot of predicted prices
                plot_url = generate_prediction_plot(predicted_prices, coin)

                # Display the prediction result
                return render_template(
                    'prediction_result.html',
                    predicted_prices=predicted_prices,
                    coin=coin,
                    predicted_high=max(predicted_prices),
                    predicted_low=min(predicted_prices),
                    sentiment_label=sentiment_label,
                    plot_url=plot_url,
                    articles=[article['title'] for article in articles]
                )
            except Exception as e:
                logging.exception(f"An error occurred during {model_type.upper()} prediction.")
                flash("An error occurred while making the prediction. Please try again later.")
                return redirect(url_for('main_routes.prediction'))
        else:
            flash("Insufficient data for making a prediction.")
            return redirect(url_for('main_routes.prediction'))
    else:
        # Get the list of available models
        model_files = os.listdir(Config.MODEL_SAVE_PATH)
        available_models = []
        for file in model_files:
            if file.endswith('.pth') or file.endswith('.pkl'):
                # Assuming the model files are named like '<model_type>_<coin>.pth' or '.pkl'
                model_type, coin_pair_with_ext = file.split('_', 1)
                coin_pair = coin_pair_with_ext.split('.')[0]
                available_models.append(f"{model_type}_{coin_pair}")

        return render_template('prediction.html', available_models=available_models)
    
@main_routes.route('/confirm_buy', methods=['POST'])
def confirm_buy():
    coin = request.form.get('coin')
    predicted_price = float(request.form.get('predicted_price'))

    # Here, you can perform additional checks, like comparing the predicted price to the current price

    # Render a confirmation page
    return render_template(
        'confirm_buy.html',
        coin=coin,
        predicted_price=predicted_price
    )

@main_routes.route('/place_order', methods=['POST'])
def place_order():
    coin = request.form.get('coin')
    quantity = request.form.get('quantity')

    # Initialize Binance Client
    api_key = Config.BINANCE_API_KEY
    api_secret = Config.BINANCE_SECRET_KEY
    client = Client(api_key, api_secret)

    try:
        # Place a market buy order
        order = client.create_order(
            symbol=coin,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        flash(f"Order placed successfully: {order}")
        return redirect(url_for('main_routes.dashboard'))
    except Exception as e:
        logging.exception("An error occurred while placing the order.")
        flash("An error occurred while placing the order. Please try again.")
        return redirect(url_for('main_routes.dashboard'))
    

@main_routes.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Handle form submission to update settings
        # For example, update API keys or other configurations
        api_key = request.form.get('api_key')
        secret_key = request.form.get('secret_key')
        # Save the keys securely (ensure to implement secure storage)
        flash("Settings updated successfully.")
        return redirect(url_for('main_routes.dashboard'))
    else:
        return render_template('settings.html')
