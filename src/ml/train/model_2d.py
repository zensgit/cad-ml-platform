"""Lightweight 2D graph model for DXF entity graphs."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _global_mean_pool(
    x: torch.Tensor, batch: Optional[torch.Tensor]
) -> torch.Tensor:
    """Mean-pool node embeddings into per-graph embeddings.

    This mirrors the basic behavior of PyG's global_mean_pool, but keeps the
    project dependency-light.
    """

    if x.size(0) == 0:
        if batch is None:
            return x.new_zeros((1, x.size(1)))
        return x.new_zeros((0, x.size(1)))

    if batch is None:
        return x.mean(dim=0, keepdim=True)

    if batch.numel() == 0:
        return x.new_zeros((0, x.size(1)))

    num_graphs = int(batch.max().item()) + 1
    pooled = x.new_zeros((num_graphs, x.size(1)))
    pooled.index_add_(0, batch, x)

    counts = x.new_zeros((num_graphs,))
    counts.index_add_(0, batch, x.new_ones((batch.size(0),), dtype=x.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    return pooled / counts


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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.relu(self.gcn1(x, edge_index))
        x = torch.relu(self.gcn2(x, edge_index))
        pooled = _global_mean_pool(x, batch)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class EdgeSageLayer(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__()
        self.msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.self_lin = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        if edge_index.numel() == 0:
            return self.self_lin(x)
        src = edge_index[0]
        dst = edge_index[1]
        messages = self.msg(torch.cat([x[src], edge_attr], dim=1))
        agg = torch.zeros((x.size(0), messages.size(1)), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, messages)
        deg = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg = agg / deg
        return self.self_lin(x) + agg


class EdgeGraphSageClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sage1 = EdgeSageLayer(node_dim, edge_dim, hidden_dim)
        self.sage2 = EdgeSageLayer(hidden_dim, edge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.relu(self.sage1(x, edge_index, edge_attr))
        x = torch.relu(self.sage2(x, edge_index, edge_attr))
        pooled = _global_mean_pool(x, batch)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
