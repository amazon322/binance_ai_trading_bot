# models/lstm.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # For saving the scaler

logging.basicConfig(level=logging.INFO)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.input_size = input_size  # Store for saving
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,  # Add dropout to LSTM layer
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 1)  # Output size is 1 for regression

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out  # Shape: (batch_size, 1)

def train_lstm_model(
    X_train, y_train, X_val, y_val,
    epochs, batch_size, learning_rate,
    model_save_path, model_name,
    scaler,
    input_size,
    sequence_length,  # Add this parameter
    hidden_size=50,    # You can set default values or get from user input
    num_layers=2,
    dropout=0.1,
    weight_decay=0.0
):
    try:
        logging.info("Starting LSTM model training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model
        model = LSTMModel(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout
        )
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Prepare data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # Lists to store losses
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            batch_train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)  # Shape: (batch_size, 1)
                outputs = outputs.view(-1)  # Shape: (batch_size)
                y_batch = y_batch.view(-1)  # Shape: (batch_size)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                batch_train_losses.append(loss.item())

            average_train_loss = np.mean(batch_train_losses)
            train_losses.append(average_train_loss)

            # Validation step
            model.eval()
            batch_val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)  # Shape: (batch_size, 1)
                    outputs = outputs.view(-1)  # Shape: (batch_size)
                    y_batch = y_batch.view(-1)  # Shape: (batch_size)
                    loss = criterion(outputs, y_batch)
                    batch_val_losses.append(loss.item())

            average_val_loss = np.mean(batch_val_losses)
            val_losses.append(average_val_loss)

            logging.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {average_train_loss:.6f}, "
                f"Val Loss: {average_val_loss:.6f}"
            )

        # Save the trained model information
        model_info = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'sequence_length': sequence_length,  # Save sequence_length
            'dropout': dropout,
            'state_dict': model.state_dict()
        }
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model_info, os.path.join(model_save_path, model_name))

        # Save the scaler
        scaler_filename = os.path.join(model_save_path, f'scaler_{model_name.split(".")[0]}.pkl')
        joblib.dump(scaler, scaler_filename)
        logging.info(f"Scaler saved at: {scaler_filename}")

        logging.info("LSTM model training completed and saved successfully.")

        # Evaluate on validation set
        model.eval()
        val_pred = []
        y_val_actual = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)  # Shape: (batch_size, 1)
                outputs = outputs.view(-1)  # Shape: (batch_size)
                y_batch = y_batch.view(-1)  # Shape: (batch_size)

                val_pred.extend(outputs.cpu().numpy())
                y_val_actual.extend(y_batch.cpu().numpy())

        val_pred = np.array(val_pred)
        y_val_actual = np.array(y_val_actual)

        # Inverse transform predictions and actual values
        y_val_pred_inversed = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        y_val_actual_inversed = scaler.inverse_transform(y_val_actual.reshape(-1, 1)).flatten()

        # Calculate evaluation metrics
        mse = mean_squared_error(y_val_actual_inversed, y_val_pred_inversed)
        mae = mean_absolute_error(y_val_actual_inversed, y_val_pred_inversed)
        rmse = np.sqrt(mse)
        logging.info(f"Validation MSE: {mse:.6f}")
        logging.info(f"Validation MAE: {mae:.6f}")
        logging.info(f"Validation RMSE: {rmse:.6f}")

        # Optionally, plot actual vs. predicted values
        # Uncomment the following lines if you want to see the plot
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(y_val_actual_inversed, label='Actual')
        plt.plot(y_val_pred_inversed, label='Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.title('LSTM Model Actual vs Predicted Values')
        plt.legend()
        plt.show()
        """

    except Exception as e:
        logging.exception("An error occurred during LSTM model training.")
        raise e
