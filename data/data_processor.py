import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta  # Technical Analysis library
import logging
from utils.data_cache import cached

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._technical_indicators_cache = {}

    def preprocess_data(self, df):
        if df.empty:
            logger.error("DataFrame is empty.")
            return None

        # Ensure column names are in lowercase
        df.columns = df.columns.str.lower()

        # Adding technical indicators
        df = self.add_technical_indicators(df)

        # Selecting the features for training
        feature_columns = df.columns.tolist()
        data = df[feature_columns].values

        # Z-score normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized_data = (data - mean) / std

        # Store mean and std for inverse transformations if needed
        self.mean = mean
        self.std = std

        return normalized_data


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def create_sequences(self, data, sequence_length, forecast_horizon=1):
        X = []
        y = []
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            # **Shift labels to predict the next time step**
            y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon)])
        X = np.array(X)
        y = np.array(y)
        return X, y


    def split_data(self, X, y, train_size=0.8):
        split_index = int(len(X) * train_size)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        return X_train, X_val, y_train, y_val

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame with caching.
        """
        try:
            # Create cache key based on data hash
            data_hash = hash(str(data[['open', 'high', 'low', 'close', 'volume']].values.tobytes()))
            cache_key = f"technical_indicators_{data_hash}"
            
            if cache_key in self._technical_indicators_cache:
                logger.debug("Using cached technical indicators")
                return self._technical_indicators_cache[cache_key]
            
            logger.debug("Adding technical indicators.")

            # Create a copy to avoid modifying original data
            data_copy = data.copy()

            # Moving Averages - optimized with vectorized operations
            data_copy['ma5'] = data_copy['close'].rolling(window=5, min_periods=1).mean()
            data_copy['ma10'] = data_copy['close'].rolling(window=10, min_periods=1).mean()
            data_copy['ma20'] = data_copy['close'].rolling(window=20, min_periods=1).mean()

            # Exponential Moving Average
            data_copy['ema50'] = data_copy['close'].ewm(span=50, adjust=False, min_periods=1).mean()

            # Stochastic Oscillator
            stochastic = ta.momentum.StochasticOscillator(
                high=data_copy['high'], low=data_copy['low'], close=data_copy['close'], window=14
            )
            data_copy['stochastic_k'] = stochastic.stoch()

            # Relative Strength Index (RSI)
            rsi = ta.momentum.RSIIndicator(close=data_copy['close'], window=14)
            data_copy['rsi'] = rsi.rsi()

            # Moving Average Convergence Divergence (MACD)
            macd = ta.trend.MACD(close=data_copy['close'])
            data_copy['macd'] = macd.macd()
            data_copy['macd_diff'] = macd.macd_diff()
            data_copy['macd_signal'] = macd.macd_signal()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(close=data_copy['close'], window=20)
            data_copy['bb_h'] = bollinger.bollinger_hband()
            data_copy['bb_l'] = bollinger.bollinger_lband()
            
            # Bollinger Band Width
            data_copy['bb_width'] = data_copy['bb_h'] - data_copy['bb_l']

            # Rate of Change (ROC)
            data_copy['roc'] = ta.momentum.ROCIndicator(close=data_copy['close'], window=14).roc()
            
            # Close Price Difference
            data_copy['close_diff'] = data_copy['close'].diff()
            data_copy['close_diff'].fillna(0, inplace=True)

            # Percent Change in Close Price
            data_copy['percent_change_close'] = data_copy['close'].pct_change() * 100
            data_copy['percent_change_close'].fillna(0, inplace=True)

            # Handling NaN values after adding indicators - more efficient
            data_copy = data_copy.fillna(method='bfill').fillna(method='ffill')

            # Cache the result
            self._technical_indicators_cache[cache_key] = data_copy
            logger.info("Technical indicators added and cached.")
            return data_copy

        except Exception as e:
            logger.exception("Error adding technical indicators.")
            raise e
