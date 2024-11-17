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
        output = self.fc_out(output[:, -1, :])  # Taking the last output
        return output


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

        # Adjusted hyperparameters
        num_layers = 2  # Experiment with different values
        dim_model = 128  # Should be divisible by num_heads
        num_heads = 4
        dim_feedforward = 256
        dropout = 0.2  # Increased dropout to prevent overfitting

        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

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

        # Define loss function and optimizer
        criterion = nn.SmoothL1Loss()  # Huber loss
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Prepare data loaders
        # For Transformers, we need both src (input sequence) and tgt (target sequence)
        def create_tgt_sequences(X, y):
            src = X
            tgt_input = y[:, :-1]  # All but the last element
            tgt_target = y[:, 1:]  # All but the first element
            return src, tgt_input, tgt_target

        # Ensure y_train and y_val are in the correct shape
        y_train = y_train.reshape(-1, sequence_length)
        y_val = y_val.reshape(-1, sequence_length)

        # Create sequences for training
        src_train, tgt_input_train, tgt_target_train = create_tgt_sequences(X_train, y_train)
        src_val, tgt_input_val, tgt_target_val = create_tgt_sequences(X_val, y_val)

        # Convert to tensors
        src_train = torch.tensor(src_train, dtype=torch.float32)
        tgt_input_train = torch.tensor(tgt_input_train, dtype=torch.float32).unsqueeze(-1)
        tgt_target_train = torch.tensor(tgt_target_train, dtype=torch.float32).unsqueeze(-1)

        src_val = torch.tensor(src_val, dtype=torch.float32)
        tgt_input_val = torch.tensor(tgt_input_val, dtype=torch.float32).unsqueeze(-1)
        tgt_target_val = torch.tensor(tgt_target_val, dtype=torch.float32).unsqueeze(-1)

        # Prepare datasets and loaders
        train_dataset = torch.utils.data.TensorDataset(src_train, tgt_input_train, tgt_target_train)
        val_dataset = torch.utils.data.TensorDataset(src_val, tgt_input_val, tgt_target_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            batch_train_losses = []
            for src_batch, tgt_input_batch, tgt_target_batch in train_loader:
                src_batch = src_batch.to(device)
                tgt_input_batch = tgt_input_batch.to(device)
                tgt_target_batch = tgt_target_batch.to(device)

                optimizer.zero_grad()
                output = model(src_batch, tgt_input_batch)
                loss = criterion(output, tgt_target_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_train_losses.append(loss.item())

            average_train_loss = np.mean(batch_train_losses)
            train_losses.append(average_train_loss)

            # Validation step
            model.eval()
            batch_val_losses = []
            with torch.no_grad():
                for src_batch, tgt_input_batch, tgt_target_batch in val_loader:
                    src_batch = src_batch.to(device)
                    tgt_input_batch = tgt_input_batch.to(device)
                    tgt_target_batch = tgt_target_batch.to(device)

                    output = model(src_batch, tgt_input_batch)
                    loss = criterion(output, tgt_target_batch)
                    batch_val_losses.append(loss.item())

            average_val_loss = np.mean(batch_val_losses)
            val_losses.append(average_val_loss)

            # Adjust the learning rate
            scheduler.step()

            logging.info(
                f"Epoch [{epoch+1}/{epochs}], "
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
        scaler_filename = os.path.join(model_save_path, f'scaler_{model_name.split(".")[0]}.pkl')
        joblib.dump(scaler, scaler_filename)

        logging.info("Transformer model training completed and saved successfully.")

        # Optional: Plot training and validation loss over epochs
        plt.figure(figsize=(10,5))
        plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Transformer Model Training and Validation Loss')
        plt.legend()
        plt.show()

    except Exception as e:
        logging.exception("An error occurred during Transformer model training.")
        raise e
