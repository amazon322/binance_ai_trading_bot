import unittest
import numpy as np
import pandas as pd
from data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.data_processor = DataProcessor()
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            'Close': np.linspace(100, 200, 100)
        })

    def test_preprocess_data(self):
        scaled_data = self.data_processor.preprocess_data(self.df)
        # Check that scaled_data is a numpy array
        self.assertIsInstance(scaled_data, np.ndarray)
        # Check that the scaler was fitted
        self.assertIsNotNone(self.data_processor.scaler)

    def test_create_sequences(self):
        scaled_data = self.data_processor.preprocess_data(self.df)
        sequence_length = 10
        X, y = self.data_processor.create_sequences(scaled_data, sequence_length)
        # Check that X and y have the correct shapes
        self.assertEqual(X.shape[0], len(scaled_data) - sequence_length)
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(y.shape[0], len(scaled_data) - sequence_length)

    def test_split_data(self):
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 1)
        X_train, X_val, y_train, y_val = self.data_processor.split_data(X, y, train_size=0.8)
        # Check the shapes of the splits
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_val.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_val.shape[0], 20)

    def test_inverse_transform(self):
        scaled_data = self.data_processor.preprocess_data(self.df)
        original_data = self.data_processor.inverse_transform(scaled_data)
        # Check that the inverse transformed data matches the original data close enough
        np.testing.assert_array_almost_equal(original_data.flatten(), self.df['Close'].values, decimal=5)

if __name__ == '__main__':
    unittest.main()