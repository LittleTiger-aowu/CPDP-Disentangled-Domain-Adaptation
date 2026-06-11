"""
Domain discriminators for shared/private representations.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    if hidden > 0:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    return nn.Linear(in_dim, out_dim)


class SharedDomainDisc(nn.Module):
    """Domain discriminator on shared features (with GRL outside)."""

    def __init__(self, d_sh: int, num_domains: int, hidden: int = 0):
        super().__init__()
        self.net = mlp(d_sh, hidden, num_domains)

    def forward(self, z_sh):
        return self.net(z_sh)


class PrivateDomainDisc(nn.Module):
    """Domain classifier on private features (no GRL)."""

    def __init__(self, d_pr: int, num_domains: int, hidden: int = 0):
        super().__init__()
        self.net = mlp(d_pr, hidden, num_domains)

    def forward(self, z_pr):
        return self.net(z_pr)
