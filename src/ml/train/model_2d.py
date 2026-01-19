"""Lightweight 2D graph model for DXF entity graphs."""

from __future__ import annotations

import torch
from torch import nn


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if n == 0:
            return x
        device = x.device
        adj = torch.zeros((n, n), device=device, dtype=x.dtype)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        # Self-loops
        adj.fill_diagonal_(1.0)
        deg = adj.sum(dim=1).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        out = adj_norm @ x
        return self.linear(out)


class SimpleGraphClassifier(nn.Module):
    def __init__(
        self, node_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.gcn1 = SimpleGCNLayer(node_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))
        if x.size(0) == 0:
            pooled = x.new_zeros((1, x.size(1)))
        else:
            pooled = x.mean(dim=0, keepdim=True)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
