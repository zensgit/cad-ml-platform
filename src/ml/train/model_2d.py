"""Lightweight 2D graph model for DXF entity graphs."""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Maximum nodes per graph — configurable via DXF_MAX_NODES env var.
DXF_MAX_NODES_DEFAULT = 500
DXF_MAX_NODES = int(os.getenv("DXF_MAX_NODES", str(DXF_MAX_NODES_DEFAULT)))


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

    def get_encoder(self) -> "GraphEncoder":
        """Extract encoder-only (GNN layers without classification head)."""
        enc = GraphEncoder(
            node_dim=self.gcn1.linear.in_features,
            hidden_dim=self.gcn1.linear.out_features,
            model_type="gcn",
        )
        enc.gcn1.load_state_dict(self.gcn1.state_dict())
        enc.gcn2.load_state_dict(self.gcn2.state_dict())
        return enc

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained encoder weights from a contrastive checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", {}))
        own = self.state_dict()
        loaded = 0
        for k, v in encoder_state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own)
        return loaded


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

    def get_encoder(self) -> "GraphEncoder":
        """Extract encoder-only (GNN layers without classification head)."""
        enc = GraphEncoder(
            node_dim=self.sage1.self_lin.in_features,
            edge_dim=self.sage1.msg.in_features - self.sage1.self_lin.in_features,
            hidden_dim=self.sage1.self_lin.out_features,
            model_type="edge_sage",
        )
        enc.sage1.load_state_dict(self.sage1.state_dict())
        enc.sage2.load_state_dict(self.sage2.state_dict())
        return enc

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained encoder weights from a contrastive checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", {}))
        # Load only the GNN layers (sage1, sage2), skip classifier
        own = self.state_dict()
        loaded = 0
        for k, v in encoder_state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own)
        return loaded


# --------------------------------------------------------------------------- #
# GraphEncoder — GNN layers only (no classification head).
# Used for contrastive pretraining and as a reusable backbone.
# --------------------------------------------------------------------------- #


class GraphEncoder(nn.Module):
    """GNN encoder that produces graph-level embeddings without a classifier.

    Supports both GCN and EdgeGraphSage architectures.  The output is a
    ``[B, hidden_dim]`` embedding after global mean-pooling.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        edge_dim: int = 0,
        model_type: str = "edge_sage",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        if model_type == "edge_sage":
            self.sage1 = EdgeSageLayer(node_dim, edge_dim, hidden_dim)
            self.sage2 = EdgeSageLayer(hidden_dim, edge_dim, hidden_dim)
        else:
            self.gcn1 = SimpleGCNLayer(node_dim, hidden_dim)
            self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_type == "edge_sage":
            if edge_attr is None:
                raise ValueError("edge_attr required for edge_sage encoder")
            x = torch.relu(self.sage1(x, edge_index, edge_attr))
            x = torch.relu(self.sage2(x, edge_index, edge_attr))
        else:
            x = torch.relu(self.gcn1(x, edge_index))
            x = torch.relu(self.gcn2(x, edge_index))
        pooled = _global_mean_pool(x, batch)
        return self.dropout(pooled)


def _attention_pool(
    x: torch.Tensor,
    gate: nn.Linear,
    batch: Optional[torch.Tensor],
) -> torch.Tensor:
    """Attention pooling: weighted sum of node embeddings per graph.

    gate: Linear(hidden_dim → 1) that scores each node.
    Produces [num_graphs, hidden_dim] output.
    """
    if x.size(0) == 0:
        if batch is None:
            return x.new_zeros((1, x.size(1)))
        return x.new_zeros((0, x.size(1)))

    scores = gate(x)  # [N, 1]

    if batch is None:
        attn = torch.softmax(scores, dim=0)  # [N, 1]
        return (attn * x).sum(dim=0, keepdim=True)  # [1, D]

    num_graphs = int(batch.max().item()) + 1
    # Per-graph softmax via subtract-max trick for stability
    max_scores = x.new_zeros((num_graphs, 1))
    max_scores.index_reduce_(0, batch, scores, reduce="amax", include_self=True)
    shifted = scores - max_scores[batch]
    exp_s = torch.exp(shifted)
    sum_exp = x.new_zeros((num_graphs, 1))
    sum_exp.index_add_(0, batch, exp_s)
    attn = exp_s / (sum_exp[batch] + 1e-9)  # [N, 1]

    pooled = x.new_zeros((num_graphs, x.size(1)))
    pooled.index_add_(0, batch, attn * x)
    return pooled


class GraphEncoderV2(nn.Module):
    """3-layer EdgeGraphSAGE encoder with residual connections and attention pooling.

    Upgrades over GraphEncoder:
    - 3 EdgeSageLayer (vs 2)
    - Residual connections between layers
    - LayerNorm after each layer
    - Attention pooling (vs mean pooling)
    - Larger hidden_dim (default 256)
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        edge_dim: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection (node_dim → hidden_dim for residual alignment)
        self.lin_in = nn.Linear(node_dim, hidden_dim)

        # 3 EdgeSage layers (all operate on hidden_dim)
        self.sage1 = EdgeSageLayer(hidden_dim, edge_dim, hidden_dim)
        self.sage2 = EdgeSageLayer(hidden_dim, edge_dim, hidden_dim)
        self.sage3 = EdgeSageLayer(hidden_dim, edge_dim, hidden_dim)

        # LayerNorm per layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Attention pooling gate
        self.gate = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr required for GraphEncoderV2")

        # Project input to hidden_dim
        h = torch.relu(self.lin_in(x))  # [N, hidden_dim]

        # Layer 1 + residual
        h1 = torch.relu(self.norm1(self.sage1(h, edge_index, edge_attr)))
        h1 = self.dropout(h1) + h  # residual

        # Layer 2 + residual
        h2 = torch.relu(self.norm2(self.sage2(h1, edge_index, edge_attr)))
        h2 = self.dropout(h2) + h1  # residual

        # Layer 3 (no residual — final representation)
        h3 = torch.relu(self.norm3(self.sage3(h2, edge_index, edge_attr)))
        h3 = self.dropout(h3)

        # Attention pooling → graph-level embedding [num_graphs, hidden_dim]
        return _attention_pool(h3, self.gate, batch)


class GraphEncoderV2WithHead(nn.Module):
    """GraphEncoderV2 + Linear classifier head — unified module for inference.

    Wraps the separately-trained encoder and classifier into one module so
    that ``vision_2d.Graph2DClassifier`` can call it with the same interface
    as ``EdgeGraphSageClassifier``.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoderV2(node_dim, hidden_dim, edge_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.encoder(x, edge_index, edge_attr=edge_attr, batch=batch)
        return self.classifier(emb)

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "GraphEncoderV2WithHead":
        """Reconstruct from a B4.4-style checkpoint dict."""
        label_map = checkpoint["label_map"]
        num_classes = checkpoint.get("num_classes", len(label_map))
        hidden_dim = checkpoint.get("hidden_dim", 256)
        node_dim = checkpoint.get("node_dim", 19)
        edge_dim = checkpoint.get("edge_dim", 7)
        model = cls(node_dim, edge_dim, hidden_dim, num_classes)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        return model


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Normalized Temperature-scaled Cross-Entropy (NT-Xent / InfoNCE) loss.

    Args:
        z1: Projected embeddings for view 1, shape ``[B, D]``.
        z2: Projected embeddings for view 2, shape ``[B, D]``.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    batch_size = z1.size(0)
    if batch_size == 0:
        return z1.new_tensor(0.0)

    # L2-normalize
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    # Concatenate: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.T) / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = nn.functional.cross_entropy(sim, labels)
    return loss
