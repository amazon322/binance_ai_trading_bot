from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import Aer
from qiskit.circuit import Parameter
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import torch.optim as optim
from qiskit.primitives import BitArray, StatevectorSampler
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from binance.client import Client

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_predictions(y_true, y_pred, title='QLSTM Predictions vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_qubits=4, num_layers=2):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Classical LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)

        # Add trainable parameters for the quantum circuits
        self.theta_f = nn.Parameter(torch.randn(num_qubits))
        self.theta_i = nn.Parameter(torch.randn(num_qubits))
        self.theta_u = nn.Parameter(torch.randn(num_qubits))
        self.theta_o = nn.Parameter(torch.randn(num_qubits))

        # Add classical linear layers
        self.linear_ih = nn.Linear(input_size, hidden_size)
        self.linear_hh = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)

        # Add batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        # Initialize VQCs for LSTM gates
        self.vqc_forget = self._create_vqc()
        self.vqc_input = self._create_vqc()
        self.vqc_update = self._create_vqc()
        self.vqc_output = self._create_vqc()

        # Add projection layer
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        h_t, c_t = hidden
        f_t = self._execute_vqc(self.vqc_forget, x, h_t)
        i_t = self._execute_vqc(self.vqc_input, x, h_t)
        g_t = self._execute_vqc(self.vqc_update, x, h_t)
        o_t = self._execute_vqc(self.vqc_output, x, h_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        output = self.projection(h_t)  # Project to correct dimension
        return output.view(x.size(0), -1), (h_t, c_t)  # Ensure output is 2D with correct batch size

    def _execute_vqc(self, qc, x, h):
        """Execute a variational quantum circuit"""
        # Get parameters from input and hidden state and reshape
        params = torch.cat([x.reshape(-1), h.reshape(-1)])
        params_np = params.detach().numpy()

        # Create parameter dictionary with correct dimensions
        parameter_dict = {}
        for i, param in enumerate(qc.parameters):
            if i < len(params_np):
                parameter_dict[param] = float(params_np[i])

        # Bind parameters to circuit
        bound_circuit = qc.assign_parameters(parameter_dict)

        # Execute circuit
        sampler = StatevectorSampler()
        job = sampler.run([(bound_circuit, [])])
        result = job.result()

        # Get measurement results from DataBin
        data = result[0].data

        # Access the BitArray data
        bitstrings = None
        for key, value in data.items():
            if isinstance(value, BitArray):
                bitstrings = value.get_bitstrings()
                break

        if bitstrings is None:
            print("No BitArray found in DataBin.")
            raise ValueError("No BitArray found in DataBin.")

        # Convert bitstrings to integers
        counts = {bitstring: int(bitstring, 2) for bitstring in bitstrings}
          # Calculate expectation value from measurement outcomes
        total_counts = sum(counts.values())
        if total_counts == 0:
            print("Total counts are zero, cannot calculate expectation value.")
            raise ValueError("Total counts are zero, cannot calculate expectation value.")

        expectation = sum(int(state, 2) * count / total_counts for state, count in counts.items())

        return torch.tensor(expectation, device=x.device)

    def _create_vqc(self):
        qc = QuantumCircuit(self.num_qubits)
        # Simple circuit with a Hadamard gate on each qubit
        for i in range(self.num_qubits):
            qc.h(i)
        qc.measure_all()
        return qc
    def forward(self, x, hidden):
        h_t, c_t = hidden
        f_t = self._execute_vqc(self.vqc_forget, x, h_t)
        i_t = self._execute_vqc(self.vqc_input, x, h_t)
        g_t = self._execute_vqc(self.vqc_update, x, h_t)
        o_t = self._execute_vqc(self.vqc_output, x, h_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

def train_qlstm_model(
    X_train, y_train, X_val, y_val,
    epochs, batch_size, learning_rate,
    model_save_path, model_name,
    sequence_length,
    scaler,
    input_size,
    hidden_size=128,
    num_qubits=4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QLSTM(input_size, hidden_size, num_qubits).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize hidden state with correct dimensions
    num_layers = 1  # Assuming single layer LSTM
    h_0 = torch.zeros(num_layers, X_train.size(0), hidden_size).to(device)
    c_0 = torch.zeros(num_layers, X_train.size(0), hidden_size).to(device)
    hidden = (h_0, c_0)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass through QLSTM with hidden state
        h_t, _ = model(X_train, hidden)
        
        # Ensure h_t and y_train have the same shape
        h_t = h_t.view(y_train.size(0), -1)

        loss = criterion(h_t, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_hidden = (torch.zeros(num_layers, X_val.size(0), hidden_size).to(device),
                          torch.zeros(num_layers, X_val.size(0), hidden_size).to(device))
            val_h_t, _ = model(X_val, val_hidden)
            val_h_t = val_h_t.view(y_val.size(0), -1)
            val_loss = criterion(val_h_t, y_val)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

def create_sequences(data, sequence_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def predict_future_prices(model, last_sequence, scaler, n_steps=10):
    model.eval()
    future_predictions = []
    current_sequence = last_sequence.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            # Initialize hidden state
            batch_size = current_sequence.size(0)
            hidden_size = model.hidden_size
            device = next(model.parameters()).device
            hidden = (
                torch.zeros(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device)
            )

            # Get prediction
            prediction, _ = model(current_sequence, hidden)
            future_predictions.append(prediction.cpu().numpy())

            # Update sequence
            current_sequence = torch.roll(current_sequence, -1, dims=1)
            current_sequence[:, -1] = prediction.squeeze()

    # Inverse transform predictions
    future_predictions = np.array(future_predictions).squeeze()
    future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

    return future_predictions

def predict_qlstm(model, X_test, scaler):
    model.eval()
    with torch.no_grad():
        # Initialize hidden state for prediction with correct dimensions
        num_layers = model.num_layers
        batch_size = X_test.size(0)
        hidden_size = model.hidden_size
        device = next(model.parameters()).device

        h_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        c_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        hidden = (h_0, c_0)

        # Make predictions with hidden state
        h_t, _ = model(X_test, hidden)

        # Reshape predictions to 2D
        predictions = h_t.view(-1, 1).cpu().numpy()

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)
    return predictions


def test_qlstm_model():
    # Initialize DataLoader
    data_loader = DataLoader()

    # Get historical data (e.g., last 30 days of BTCUSDT)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    df = data_loader.get_historical_data(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str=start_str,
        end_str=end_str
    )

    # Prepare data
    data_series = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series)

    # Create sequences
    sequence_length = 60
    X, y = create_sequences(scaled_data, sequence_length)

    # Split data
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Initialize and train QLSTM
    input_size = 1
    hidden_size = 256
    num_qubits = 8
    model = QLSTM(input_size, hidden_size, num_qubits)

    # Train the model
    train_qlstm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=50,
        batch_size=64,
        learning_rate=0.01,
        model_save_path='models/saved',
        model_name='qlstm_btc.pth',
        sequence_length=sequence_length,
        scaler=scaler,
        input_size=input_size,
        hidden_size=hidden_size,
        num_qubits=num_qubits
    )

    # Make predictions
    predictions = predict_qlstm(model, X_test, scaler)

    # Convert tensors to numpy arrays for metric calculation
    y_test_np = y_test.detach().numpy()
    predictions_np = predictions  # Already a numpy array from scaler.inverse_transform

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_np, predictions_np))
    accuracy = r2_score(y_test_np, predictions_np)

    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Plot actual vs predicted
    plot_predictions(y_test_np, predictions_np)

    # Predict future prices
    future_prices = predict_future_prices(
        model=model,
        last_sequence=X_test[-1:],  # Use last sequence from test data
        scaler=scaler,
        n_steps=10  # Predict next 10 time steps
    )

    print("\nPredicted future prices:")
    for i, price in enumerate(future_prices):
        print(f"Step {i+1}: {price[0]:.2f}")

    return model, predictions, future_prices

def train_step(model, batch_X, batch_y, hidden, optimizer, criterion):
    optimizer.zero_grad()
    output, hidden = model(batch_X, hidden)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    return loss, hidden

if __name__ == '__main__':
    model, predictions, future_prices = test_qlstm_model()
