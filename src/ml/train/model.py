"""
UV-Net Graph Model Definition (Dual-Path Implementation).

Defines the architecture for the 3D feature recognition model.
Adheres to the "Graph Data Contract" v1.0.

### Graph Data Contract (v1.0)
Input is a Data object (or named tuple) with:
- x: Tensor[N, node_input_dim] (float32) - Node features (Face attributes)
- edge_index: Tensor[2, E] (int64) - Adjacency list (Source, Target)
- edge_attr: Tensor[E, edge_input_dim] (float32) - Edge features (optional)
- batch: Tensor[N] (int64) - Batch index for each node (0..batch_size-1)

This module supports two backends:
1. `torch_geometric` (Preferred): Optimized GNN kernels.
2. `pure_torch` (Fallback): Basic matrix multiplication GCN for environments without PyG.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --- Dependency Management ---
HAS_PYG = False
HAS_PYG_EDGE_ATTR = False
try:
    from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool

    HAS_PYG = True
    HAS_PYG_EDGE_ATTR = True
except ImportError:
    logger.info("torch_geometric not found. Using pure PyTorch fallback for GNN layers.")


# --- Pure PyTorch Fallback Layer ---
class SimpleGCNLayer(nn.Module):
    """
    Basic Graph Convolutional Layer using pure PyTorch.
    Formula: D^-0.5 * A * D^-0.5 * X * W
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [N, in_channels]
        # edge_index: [2, E]
        N = x.size(0)
        device = x.device

        # 1. Self-loops (add identity matrix)
        # Simply append 0..N-1 to edge_index
        loop_index = torch.arange(0, N, dtype=torch.long, device=device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index_aug = torch.cat([edge_index, loop_index], dim=1)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(device).view(-1)
        loop_weight = torch.ones(N, device=device, dtype=edge_weight.dtype)
        edge_weight_aug = torch.cat([edge_weight, loop_weight], dim=0)

        # 2. Compute Adjacency Matrix (Sparse)
        # Value is 1.0 for all edges
        row, col = edge_index_aug
        if x.device.type == "mps":
            adj = torch.zeros((N, N), device=device, dtype=edge_weight_aug.dtype)
            adj.index_put_((row, col), edge_weight_aug, accumulate=True)
        else:
            adj = torch.sparse_coo_tensor(
                torch.stack([row, col]), edge_weight_aug, (N, N)
            ).to_dense()  # Warning: O(N^2) memory, simplistic for fallback

        # 3. Compute Degree Matrix D
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        d_mat_inv_sqrt = torch.diag(deg_inv_sqrt)

        # 4. Propagation: D^-0.5 * A * D^-0.5 * X
        support = torch.mm(d_mat_inv_sqrt, adj)
        support = torch.mm(support, d_mat_inv_sqrt)
        output = torch.mm(support, x)

        # 5. Weights
        output = self.linear(output) + self.bias
        return output


def simple_global_max_pool(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Global max pooling for pure torch batching."""
    if batch is None:
        return x.max(dim=0, keepdim=True)[0]

    # Number of graphs in batch
    batch_size = batch.max().item() + 1
    out_list = []
    for i in range(batch_size):
        mask = batch == i
        if mask.sum() > 0:
            out_list.append(x[mask].max(dim=0)[0])
        else:
            out_list.append(torch.zeros(x.size(1), device=x.device))
    return torch.stack(out_list)


# --- Main Model Architecture ---
class UVNetGraphModel(nn.Module):
    """
    GNN for CAD B-Rep classification.
    """

    def __init__(
        self,
        node_input_dim: Optional[int] = None,
        edge_input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        embedding_dim: int = 1024,
        num_classes: int = 11,
        dropout_rate: float = 0.3,
        node_schema: Optional[Tuple[str, ...]] = None,
        edge_schema: Optional[Tuple[str, ...]] = None,
        use_edge_attr: bool = True,
    ):
        super().__init__()
        if node_input_dim is None:
            if node_schema is not None:
                node_input_dim = len(node_schema)
            else:
                node_input_dim = 15
        if edge_input_dim is None:
            if edge_schema is not None:
                edge_input_dim = len(edge_schema)
            else:
                edge_input_dim = 2
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.node_schema = tuple(node_schema) if node_schema else None
        self.edge_schema = tuple(edge_schema) if edge_schema else None
        self.has_pyg = HAS_PYG
        self.edge_weight_layer = nn.Linear(edge_input_dim, 1) if edge_input_dim > 0 else None
        self.use_edge_attr = use_edge_attr
        self.edge_mlp = None
        self._use_gine = False

        # Encoder Layers (GCN)
        if self.has_pyg:
            if self.use_edge_attr and HAS_PYG_EDGE_ATTR:
                self.edge_mlp = nn.Sequential(
                    nn.Linear(edge_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.conv1 = GINEConv(
                    nn.Sequential(
                        nn.Linear(node_input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=edge_input_dim,
                )
                self.conv2 = GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    ),
                    edge_dim=edge_input_dim,
                )
                self.conv3 = GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, embedding_dim),
                        nn.ReLU(),
                        nn.Linear(embedding_dim, embedding_dim),
                    ),
                    edge_dim=edge_input_dim,
                )
                self._use_gine = True
            else:
                self.conv1 = GCNConv(node_input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
                self.conv3 = GCNConv(hidden_dim * 2, embedding_dim)
        else:
            self.conv1 = SimpleGCNLayer(node_input_dim, hidden_dim)
            self.conv2 = SimpleGCNLayer(hidden_dim, hidden_dim * 2)
            self.conv3 = SimpleGCNLayer(hidden_dim * 2, embedding_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

        # Classification Head (MLP)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [Num_Nodes, Node_Features]
            edge_index: [2, Num_Edges]
            batch: [Num_Nodes] - Batch assignments (0..B-1)

        Returns:
            logits: [Batch_Size, Num_Classes]
            embedding: [Batch_Size, Embedding_Dim]
        """
        edge_weight = self._edge_weight(edge_attr)

        # 1. Graph Convolution Blocks
        if self.has_pyg and self._use_gine:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        if self.has_pyg and self._use_gine:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        if self.has_pyg and self._use_gine:
            x = self.conv3(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        # Activation optional here before pooling, usually helpful
        x = F.relu(x)

        # 2. Global Pooling (Readout)
        if self.has_pyg:
            # PyG handles masking internally
            embedding = global_max_pool(x, batch)
        else:
            embedding = simple_global_max_pool(x, batch)

        # 3. Classification Head
        out = self.fc1(embedding)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)

        logits = self.fc3(out)

        return F.log_softmax(logits, dim=1), embedding

    def get_config(self):
        """Return architecture configuration for saving."""
        return {
            "node_input_dim": self.node_input_dim,
            "edge_input_dim": self.edge_input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "node_schema": self.node_schema,
            "edge_schema": self.edge_schema,
            "use_edge_attr": self.use_edge_attr,
            "backend": "pyg" if self.has_pyg else "pure_torch",
            "edge_backend": "gine" if self._use_gine else "weighted_gcn",
        }

    def _edge_weight(self, edge_attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return None
        if self.edge_weight_layer is None:
            return None
        if not self.use_edge_attr:
            return None
        if edge_attr.dim() != 2:
            raise ValueError(f"Edge feature tensor must be 2D, got shape {tuple(edge_attr.shape)}")
        if edge_attr.size(1) != self.edge_input_dim:
            raise ValueError(
                f"Edge feature dim mismatch: expected {self.edge_input_dim}, got {edge_attr.size(1)}"
            )
        edge_attr = edge_attr.to(dtype=self.edge_weight_layer.weight.dtype)
        return torch.sigmoid(self.edge_weight_layer(edge_attr)).squeeze(-1)
