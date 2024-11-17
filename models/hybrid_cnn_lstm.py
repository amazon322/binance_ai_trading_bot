import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

logging.basicConfig(level=logging.INFO)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2,dropout=0.0):
        super(CNNLSTMModel, self).__init__()
        self.input_size = input_size  # Store for saving
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Ensure dropout is a float
        self.dropout_prob = dropout  # Store dropout probability
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=16,
            kernel_size=3
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Shape: [batch_size, input_size, sequence_length]
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Shape: [batch_size, sequence_length_reduced, features]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Shape: [batch_size, 1]
        return out

def train_cnn_lstm_model(
    X_train, y_train, X_val, y_val,
    epochs, batch_size, learning_rate,
    model_save_path, model_name,
    scaler, sequence_length
):
    try:
        logging.info("Starting CNN-LSTM model training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = X_train.shape[2]  # Number of features

        # Initialize the model
        model = CNNLSTMModel(input_size=input_size)
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Prepare data loaders
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train), torch.Tensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val), torch.Tensor(y_val)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )

        # Lists to store losses
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            batch_train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                outputs = outputs.view(-1)
                y_batch = y_batch.view(-1)

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
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(X_batch)
                    outputs = outputs.view(-1)
                    y_batch = y_batch.view(-1)

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
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'sequence_length': sequence_length,
             'dropout': model.dropout_prob,
            'state_dict': model.state_dict()
        }
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model_info, os.path.join(model_save_path, model_name))

        # Save the scaler
        scaler_filename = os.path.join(model_save_path, f'scaler_{model_name.split(".")[0]}.pkl')
        joblib.dump(scaler, scaler_filename)

        logging.info("CNN-LSTM model training completed and saved successfully.")

        # Plot training and validation loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('CNN-LSTM Model Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        # Save the plot
        loss_plot_path = os.path.join(model_save_path, f'cnn_lstm_training_loss_{model_name.split("_")[1].split(".")[0]}.png')
        plt.savefig(loss_plot_path)
        plt.close()

        logging.info(f"Training loss plot saved to {loss_plot_path}")

    except Exception as e:
        logging.exception("An error occurred during CNN-LSTM model training.")
        raise e
