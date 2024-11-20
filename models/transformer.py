# models/transformer.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Configure logging
logging.basicConfig(level=logging.INFO)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(1, max_len, dim_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to prevent updates during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, dim_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        dim_model=256,
        num_heads=16,
        num_encoder_layers=12,
        dim_feedforward=1024,
        dropout=0.2,
        activation="relu"
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.dim_model = dim_model

        # Input projection layer
        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Add this parameter
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        # Output projection layer
        self.fc_out = nn.Linear(dim_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_size)
        src = self.embedding(src)  # (batch_size, sequence_length, dim_model)
        src = self.pos_encoder(src)  # (batch_size, sequence_length, dim_model)
        # No need to permute when batch_first=True
        output = self.transformer_encoder(src)  # (batch_size, sequence_length, dim_model)
        output = output[:, -1, :]  # Take the output corresponding to the last time step
        output = self.fc_out(output)  # (batch_size, 1)
        return output

def train_transformer_model(
    X_train, y_train, X_val, y_val,
    epochs, batch_size, learning_rate,
    model_save_path, model_name,
    scaler,
    sequence_length,
    input_size,
    hidden_size=256,
    num_heads=16,
    num_encoder_layers=12,
    dim_feedforward=1024,
    dropout=0.2,
    device=None
):
    try:
        logging.info("Starting Transformer model training...")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Ensure y_train and y_val are 1D tensors
        y_train = y_train.view(-1)
        y_val = y_val.view(-1)

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize the model
        model = TimeSeriesTransformer(
            input_size=input_size,
            dim_model=hidden_size,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        model.to(device)

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Initialize learning rate scheduler
        scheduler = get_lr_scheduler(optimizer, epochs)

        # Variables for early stopping and best model saving
        best_val_loss = float('inf')
        patience = 10
        counter = 0

        # Training loop
        for epoch in range(epochs):
            model.train()
            batch_train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)  # outputs shape: (batch_size, 1)
                outputs = outputs.view(-1)  # reshape to (batch_size)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                batch_train_losses.append(loss.item())

            average_train_loss = np.mean(batch_train_losses)

            # Validation loop
            model.eval()
            batch_val_losses = []
            correct_direction = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(X_batch)
                    outputs = outputs.view(-1)

                    loss = criterion(outputs, y_batch)
                    batch_val_losses.append(loss.item())

                    # Directional accuracy
                    y_prev = X_batch[:, -1, 0]  # Assuming the first feature is the target variable
                    y_true_change = y_batch - y_prev
                    y_pred_change = outputs - y_prev
                    direction = (torch.sign(y_true_change) == torch.sign(y_pred_change)).float()
                    correct_direction.extend(direction.cpu().numpy())

            average_val_loss = np.mean(batch_val_losses)
            directional_accuracy = np.mean(correct_direction)

            # Scheduler step
            scheduler.step()

            logging.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {average_train_loss:.6f}, "
                f"Val Loss: {average_val_loss:.6f}, "
                f"Directional Accuracy: {directional_accuracy:.2%}"
            )

            # Check for improvement
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                counter = 0
                # Save the best model
                os.makedirs(model_save_path, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_size': model.input_size,
                    'dim_model': model.dim_model,
                    'num_heads': num_heads,
                    'num_encoder_layers': num_encoder_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'sequence_length': sequence_length
                }, os.path.join(model_save_path, f'best_{model_name}'))
                logging.info(f"Saved best model at epoch {epoch+1}")
            else:
                counter += 1
                if counter >= patience:
                    logging.info("Early stopping triggered")
                    break

        # Load the best model after training
        best_model_path = os.path.join(model_save_path, f'best_{model_name}')
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': model.input_size,
            'dim_model': model.dim_model,
            'num_heads': num_heads,
            'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'sequence_length': sequence_length
        }, os.path.join(model_save_path, model_name))
        logging.info("Transformer model training completed and saved successfully.")

        # Save the scaler
        scaler_filename = os.path.join(model_save_path, f'scaler_transformer_{model_name}.pkl')
        import joblib
        joblib.dump(scaler, scaler_filename)

    except Exception as e:
        logging.exception("An error occurred during Transformer model training.")
        raise e

def get_lr_scheduler(optimizer, epochs, lr_start=0.0001, lr_max=0.005, lr_min=0.00001, lr_ramp_ep=30, lr_sus_ep=0):
    def lr_lambda(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(phase))
        return lr / lr_max  # Normalize by max learning rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def evaluate_transformer_model(model, test_loader, device):
    model.eval()
    y_true_list = []
    y_pred_list = []
    correct_direction = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            outputs = outputs.view(-1)

            y_true_list.extend(y_batch.cpu().numpy())
            y_pred_list.extend(outputs.cpu().numpy())

            # Directional accuracy
            y_prev = X_batch[:, -1, 0]
            y_true_change = y_batch - y_prev
            y_pred_change = outputs - y_prev
            direction = (torch.sign(y_true_change) == torch.sign(y_pred_change)).float()
            correct_direction.extend(direction.cpu().numpy())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    directional_accuracy = np.mean(correct_direction)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    logging.info(f"Test MSE: {mse:.6f}")
    logging.info(f"Test MAE: {mae:.6f}")
    logging.info(f"Test RMSE: {rmse:.6f}")
    logging.info(f"Test R-squared: {r2:.6f}")
    logging.info(f"Directional Accuracy: {directional_accuracy:.2%}")

    return mse, mae, rmse, r2, directional_accuracy
