import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from data.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    @patch('data.data_loader.BinanceAPI')
    def test_get_historical_data_success(self, mock_binance_api):
        # Correct mock data setup
        mock_row = [
            1625097600000, "34000.00", "35000.00", "33000.00", "34500.00",
            "100.0", 1625101200000, "3450000.0", 1000, "50.0", "1725000.0", "0"
        ]
        mock_klines = [mock_row.copy() for _ in range(60)]

        mock_binance_api.return_value.get_historical_klines.return_value = mock_klines

        data_loader = DataLoader()
        df = data_loader.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_str='1 Jan 2021',
            end_str='2 Jan 2021'
        )

        # Check that the DataFrame is returned and has the correct length
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 60)


    @patch('data.data_loader.BinanceAPI')
    def test_get_historical_data_no_data(self, mock_binance_api):
        # Mock the BinanceAPI's get_historical_klines method to return empty list
        mock_binance_api.return_value.get_historical_klines.return_value = []

        data_loader = DataLoader()
        df = data_loader.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_str='1 Jan 2021',
            end_str='2 Jan 2021'
        )

        # Check that None is returned when no data is retrieved
        self.assertIsNone(df)

    @patch('data.data_loader.BinanceAPI')
    def test_get_historical_data_exception(self, mock_binance_api):
        # Mock the BinanceAPI's get_historical_klines method to raise an exception
        mock_binance_api.return_value.get_historical_klines.side_effect = Exception('API error')

        data_loader = DataLoader()
        df = data_loader.get_historical_data(
            symbol='BTCUSDT',
            interval='1h',
            start_str='1 Jan 2021',
            end_str='2 Jan 2021'
        )

        # Check that None is returned when an exception occurs
        self.assertIsNone(df)

if __name__ == '__main__':
    unittest.main()