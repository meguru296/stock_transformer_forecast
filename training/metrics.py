import numpy as np
import torch
from scipy.stats import spearmanr

def compute_metrics(pred, target):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    ic = np.corrcoef(pred, target)[0, 1] if len(pred) > 1 else 0
    rank_ic, _ = spearmanr(pred, target) if len(pred) > 1 else (0, 1)
    direction_acc = np.mean((pred > 0) == (target > 0))

    return {
        "mse": float(mse),
        "mae": float(mae),
        "ic": float(ic),
        "rank_ic": float(rank_ic),
        "direction_acc": float(direction_acc)
    }
