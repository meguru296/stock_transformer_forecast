import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta)
    def forward(self, pred, target, mask=None):
        return self.huber(pred, target)
