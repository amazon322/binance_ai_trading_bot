import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .tcn import TCN

class HybridModel(nn.Module):
    def __init__(self, in_features, tcn_channels=(32,64,64), lstm_hidden=64, nhead=4, tf_layers=2, num_classes=3):
        super().__init__()
        self.tcn = TCN(in_features, tcn_channels)
        self.bilstm = nn.LSTM(input_size=tcn_channels[-1], hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        d_model = lstm_hidden*2
        enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.tf = TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.reg_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        z = self.tcn(x)
        z, _ = self.bilstm(z)
        z = self.tf(z)
        z_last = z[:, -1, :]
        logits = self.cls_head(z_last)
        yreg = self.reg_head(z_last).squeeze(-1)
        return logits, yreg