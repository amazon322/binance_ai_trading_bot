# Placeholder for dataset utilities
# This file can be extended with data loading and preprocessing utilities
# for training the hybrid model

import torch
from torch.utils.data import Dataset, DataLoader

class FuturesDataset(Dataset):
    """Placeholder dataset class for futures data"""
    def __init__(self, X, y_cls, y_reg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_cls[idx], self.y_reg[idx]