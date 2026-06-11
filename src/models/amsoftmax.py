"""
AM-Softmax head for binary classification (can be extended to multi-class).
Implements margin and scale on normalized features and weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, margin: float = 0.35, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(in_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        # x: [B, d], labels: [B]
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=0)
        logits = torch.matmul(x_norm, w_norm)  # [B, C]
        if labels is not None:
            one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
            logits = logits - one_hot * self.margin
        return logits * self.scale
