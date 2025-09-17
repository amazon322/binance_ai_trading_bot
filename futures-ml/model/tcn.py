import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_ch, channels=(32,64,64), k=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_c in enumerate(channels):
            dilation = 2 ** i
            in_c = in_ch if i == 0 else channels[i-1]
            pad = (k - 1) * dilation
            layers.append(TemporalBlock(in_c, out_c, k, 1, dilation, pad, dropout))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x = x.transpose(1,2)
        y = self.network(x)
        return y.transpose(1,2)