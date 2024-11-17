# models/transformer.py

import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import logging
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import matplotlib.pyplot as plt

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
    def __init__(self, input_size, num_encoder_layers, num_decoder_layers, dim_model, num_heads, dim_feedforward, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_size
        self.dim_model = dim_model

        self.embedding = nn.Linear(input_size, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: set batch_first=True to align dimensions
        )
        self.fc_out = nn.Linear(dim_model, 1)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)  # Shape: [batch_size, src_seq_len, dim_model]
        tgt_emb = self.embedding(tgt)  # Shape: [batch_size, tgt_seq_len, dim_model]
        
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer(src_emb, tgt_emb)
        output = self.fc_out(output)  # Shape: [batch_size, tgt_seq_len, output_size]
        return output


    # models/transformer.py

def train_transformer_model(
    X_train, y_train, X_val, y_val,
    epochs, batch_size, learning_rate,
    model_save_path, model_name,
    scaler
):
    try:
        logging.info("Starting Transformer model training...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_size = X_train.shape[2]
        sequence_length = X_train.shape[1]

        # Transformer hyperparameters
        num_layers = 2
        dim_model = 128
        num_heads = 4
        dim_feedforward = 256
        dropout = 0.2

        # Initialize the model
        model = TimeSeriesTransformer(
            input_size=input_size,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(device)

        # Function to create target sequences
        def create_tgt_sequences(y, seq_len):
            tgt_input = []
            tgt_target = []
            for i in range(len(y) - seq_len):
                tgt_input.append(y[i:i + seq_len])
                tgt_target.append(y[i + 1:i + seq_len + 1])
            return np.array(tgt_input), np.array(tgt_target)

        # Flatten y_train and y_val
        y_train_flat = y_train.flatten()
        y_val_flat = y_val.flatten()

        # Create target sequences
        tgt_input_train, tgt_target_train = create_tgt_sequences(y_train_flat, sequence_length)
        tgt_input_val, tgt_target_val = create_tgt_sequences(y_val_flat, sequence_length)

        # Adjust X_train and X_val to match the lengths of target sequences
        X_train = X_train[:len(tgt_input_train)]
        X_val = X_val[:len(tgt_input_val)]

        # Convert to tensors
        src_train = torch.tensor(X_train, dtype=torch.float32)
        tgt_input_train = torch.tensor(tgt_input_train, dtype=torch.float32).unsqueeze(-1)
        tgt_target_train = torch.tensor(tgt_target_train, dtype=torch.float32).unsqueeze(-1)

        src_val = torch.tensor(X_val, dtype=torch.float32)
        tgt_input_val = torch.tensor(tgt_input_val, dtype=torch.float32).unsqueeze(-1)
        tgt_target_val = torch.tensor(tgt_target_val, dtype=torch.float32).unsqueeze(-1)

        # Prepare datasets and loaders
        train_dataset = torch.utils.data.TensorDataset(src_train, tgt_input_train, tgt_target_train)
        val_dataset = torch.utils.data.TensorDataset(src_val, tgt_input_val, tgt_target_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for src_batch, tgt_input_batch, tgt_target_batch in train_loader:
                src_batch = src_batch.to(device)
                tgt_input_batch = tgt_input_batch.to(device)
                tgt_target_batch = tgt_target_batch.to(device)

                optimizer.zero_grad()
                output = model(src_batch, tgt_input_batch)
                loss = criterion(output, tgt_target_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation step
            model.eval()
            val_losses = []
            with torch.no_grad():
                for src_batch, tgt_input_batch, tgt_target_batch in val_loader:
                    src_batch = src_batch.to(device)
                    tgt_input_batch = tgt_input_batch.to(device)
                    tgt_target_batch = tgt_target_batch.to(device)

                    output = model(src_batch, tgt_input_batch)
                    loss = criterion(output, tgt_target_batch)
                    val_losses.append(loss.item())

            average_train_loss = np.mean(train_losses)
            average_val_loss = np.mean(val_losses)

            # Adjust the learning rate
            scheduler.step()

            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {average_train_loss:.6f}, "
                f"Val Loss: {average_val_loss:.6f}"
            )

        # Save the trained model information
        model_info = {
            "input_size": input_size,
            "sequence_length": sequence_length,
            "state_dict": model.state_dict(),
            "model_params": {
                "num_encoder_layers": num_layers,
                "num_decoder_layers": num_layers,
                "dim_model": dim_model,
                "num_heads": num_heads,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
            },
        }
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model_info, os.path.join(model_save_path, model_name))

        # Save the scaler
        scaler_filename = os.path.join(model_save_path, f'scaler_{model_name.split('.')[0]}.pkl')
        joblib.dump(scaler, scaler_filename)

        logging.info("Transformer model training completed and saved successfully.")

    except Exception as e:
        logging.exception("An error occurred during Transformer model training.")
        raise e
def create_tgt_sequences(y, seq_len):
    tgt_input = []
    tgt_target = []
    for i in range(len(y) - seq_len):
        tgt_seq = y[i:i + seq_len]
        tgt_input.append(tgt_seq[:-1])    # Length seq_len - 1
        tgt_target.append(tgt_seq[1:])    # Length seq_len - 1
    return np.array(tgt_input), np.array(tgt_target)
