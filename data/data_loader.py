from utils.binance_api import BinanceAPI
from utils.data_cache import cached
import pandas as pd
import numpy as np
import os
from scipy import stats  # For z-score calculation
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.api = BinanceAPI()

    @cached(ttl=60)  # Cache for 1 minute
    def get_historical_data(self, symbol, interval, start_str, end_str=None):
        try:
            klines = self.api.get_historical_klines(symbol, interval, start_str, end_str)
            logger.info(f"Received {len(klines)} klines for {symbol}")
            if klines:
                logger.debug(f"First kline data: {klines[0]}")
                df = self.convert_to_dataframe(klines)
                return df
            else:
                logger.warning("No data retrieved.")
                return None
        except Exception as e:
            logger.error(f"Exception occurred while fetching data: {e}")
            return None


    def convert_to_dataframe(self, klines):
        # Define column names
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(klines, columns=columns)

        # Standardize column names: convert to lowercase and remove spaces
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Convert 'timestamp' to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        

        # Convert all columns to numeric types
        df = df.apply(pd.to_numeric, errors='coerce')

        # Drop NaN and infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Remove outliers using z-score
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]

        return df

