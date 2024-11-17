# models/transformer.py

import torch
import torch.nn as nn
import math
import os
import logging
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(max_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, dim_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_model=512,  # Set to a value divisible by num_heads
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = "Transformer"
        self.input_size = input_size
        self.dim_model = dim_model

        # Embedding layer to project input features to dim_model
        self.embedding = nn.Linear(input_size, dim_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(dim_model, dropout)

        # Transformer module with batch_first=True
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Ensure batch_first is True
        )

        # Fully connected output layer
        self.fc_out = nn.Linear(dim_model, 1)

    def forward(self, src, tgt):
        # src and tgt shape: (batch_size, seq_len, input_size)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

def train_transformer_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    learning_rate,
    model_save_path,
    model_name,
    scaler,
    input_size,
    sequence_length,
    num_heads=8,
    num_layers=2,
    dim_feedforward=512,
    dropout=0.1,
    weight_decay=0.0,
    dim_model=128  # Ensure dim_model is divisible by num_heads
):
    try:
        logging.info("Starting Transformer model training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        # Initialize the model
        model = TimeSeriesTransformer(
            input_size=input_size,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Lists to store losses
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_losses = []
            for i in range(0, X_train_tensor.size(0), batch_size):
                src = X_train_tensor[i : i + batch_size]
                tgt = y_train_tensor[i : i + batch_size].unsqueeze(-1)
                tgt_input = src[:, -sequence_length // 2:, :]

                optimizer.zero_grad()
                output = model(src, tgt_input)
                loss = criterion(output[:, -1, 0], tgt.squeeze())
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for i in range(0, X_val_tensor.size(0), batch_size):
                    src = X_val_tensor[i : i + batch_size]
                    tgt = y_val_tensor[i : i + batch_size].unsqueeze(-1)
                    tgt_input = src[:, -sequence_length // 2:, :]

                    val_output = model(src, tgt_input)
                    val_loss = criterion(val_output[:, -1, 0], tgt.squeeze())
                    epoch_val_losses.append(val_loss.item())

            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)

            logging.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

        # Evaluate the model on the entire training and validation sets
        model.eval()
        with torch.no_grad():
            # Training data predictions
            src_train = X_train_tensor
            tgt_input_train = src_train[:, -sequence_length // 2:, :]
            train_pred = model(src_train, tgt_input_train)
            train_pred = train_pred[:, -1, 0].cpu().numpy()
            y_train_actual = y_train_tensor.cpu().numpy()

            # Validation data predictions
            src_val = X_val_tensor
            tgt_input_val = src_val[:, -sequence_length // 2:, :]
            val_pred = model(src_val, tgt_input_val)
            val_pred = val_pred[:, -1, 0].cpu().numpy()
            y_val_actual = y_val_tensor.cpu().numpy()

            # Inverse transform the predictions if needed
            # If y_train and y_val were scaled, inverse transform them
            y_train_actual = scaler.inverse_transform(y_train_actual.reshape(-1, 1)).flatten()
            train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            y_val_actual = scaler.inverse_transform(y_val_actual.reshape(-1, 1)).flatten()
            val_pred = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()

            # Calculate metrics
            mse_train = mean_squared_error(y_train_actual, train_pred)
            mae_train = mean_absolute_error(y_train_actual, train_pred)
            rmse_train = np.sqrt(mse_train)

            mse_val = mean_squared_error(y_val_actual, val_pred)
            mae_val = mean_absolute_error(y_val_actual, val_pred)
            rmse_val = np.sqrt(mse_val)

            print(f"\nFinal Training Metrics:")
            print(f"Train MSE: {mse_train:.4f}")
            print(f"Train MAE: {mae_train:.4f}")
            print(f"Train RMSE: {rmse_train:.4f}")

            print(f"\nFinal Validation Metrics:")
            print(f"Validation MSE: {mse_val:.4f}")
            print(f"Validation MAE: {mae_val:.4f}")
            print(f"Validation RMSE: {rmse_val:.4f}")

        # Save the trained model
        model_info = {
            "input_size": input_size,
            "sequence_length": sequence_length,
            "state_dict": model.state_dict(),
            "model_params": {
                "num_encoder_layers": num_layers,  # Use num_layers for both encoder and decoder
                "num_decoder_layers": num_layers,
                "dim_model": dim_model,
                "dim_feedforward": dim_feedforward,
                "num_heads": num_heads,
                "dropout": dropout,
                # 'batch_first' is not needed here
            },
        }

        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model_info, os.path.join(model_save_path, model_name))

        # Save the scaler
        scaler_filename = os.path.join(
            model_save_path, f"scaler_{model_name.split('.')[0]}.pkl"
        )
        joblib.dump(scaler, scaler_filename)
        logging.info("Transformer model training completed and saved successfully.")

    except Exception as e:
        logging.exception("An error occurred during Transformer model training.")
        raise e