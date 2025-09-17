import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.hybrid_model import HybridModel
from model.losses import FocalLoss

def fit(Xtr, ytr_cls, ytr_reg, Xva, yva_cls, yva_reg, epochs=10, bs=64, device="cpu"):
    model = HybridModel(in_features=Xtr.shape[-1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    dl_tr = DataLoader(TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr_cls).long(), torch.tensor(ytr_reg).float()), batch_size=bs, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.tensor(Xva).float(), torch.tensor(yva_cls).long(), torch.tensor(yva_reg).float()), batch_size=bs)
    for ep in range(1, epochs+1):
        model.train(); tot=0
        for X, ycls, yreg in dl_tr:
            X, ycls, yreg = X.to(device), ycls.to(device), yreg.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits, yhat = model(X)
                loss = FocalLoss()(logits, ycls) + 0.5*nn.SmoothL1Loss()(yhat, yreg)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        print(f"Epoch {ep}: train {tot/len(dl_tr):.4f}")
    return model