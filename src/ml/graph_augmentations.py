"""Graph augmentation strategies for contrastive learning on DXF graphs.

Implements node feature masking, edge dropout, subgraph sampling, and node
dropout.  Each augmentation returns a *new* graph dict (never modifies the
input in-place) in the same format used by ``vision_2d.py`` / ``model_2d.py``:

    {"x": Tensor[N, D], "edge_index": Tensor[2, E], "edge_attr": Tensor[E, F]}

``edge_attr`` is optional; when present it is kept consistent with edge
modifications.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _clone_graph(graph: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a deep copy of a graph dict, cloning all tensors."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------- #
# Node feature masking
# --------------------------------------------------------------------------- #

def node_feature_masking(
    graph: Dict[str, torch.Tensor],
    mask_ratio: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Randomly zero-out a fraction of node features.

    For each node, each feature dimension is independently zeroed with
    probability ``mask_ratio``.
    """
    g = _clone_graph(graph)
    x = g["x"]
    if x.numel() == 0:
        return g
    mask = torch.rand_like(x) < mask_ratio
    g["x"] = x.masked_fill(mask, 0.0)
    return g


# --------------------------------------------------------------------------- #
# Edge dropout
# --------------------------------------------------------------------------- #

def edge_dropout(
    graph: Dict[str, torch.Tensor],
    drop_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Randomly remove edges from the graph."""
    g = _clone_graph(graph)
    edge_index = g["edge_index"]
    if edge_index.numel() == 0:
        return g

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) >= drop_ratio
    g["edge_index"] = edge_index[:, keep_mask]

    if "edge_attr" in g and g["edge_attr"] is not None:
        g["edge_attr"] = g["edge_attr"][keep_mask]

    return g


# --------------------------------------------------------------------------- #
# Subgraph sampling (connected subgraph via BFS)
# --------------------------------------------------------------------------- #

def subgraph_sampling(
    graph: Dict[str, torch.Tensor],
    ratio: float = 0.8,
) -> Dict[str, torch.Tensor]:
    """Sample a connected subgraph by BFS starting from a random node.

    Keeps approximately ``ratio`` of the original nodes.  If the BFS
    component is smaller than the target, all reachable nodes are kept.
    """
    g = _clone_graph(graph)
    x = g["x"]
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return g

    target = max(1, int(num_nodes * ratio))
    edge_index = g["edge_index"]

    # Build adjacency list
    adj: Dict[int, list] = {i: [] for i in range(num_nodes)}
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)

    # BFS
    start = int(torch.randint(0, num_nodes, (1,)).item())
    visited = set()
    queue = [start]
    visited.add(start)
    while queue and len(visited) < target:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= target:
                    break

    kept = sorted(visited)
    return _reindex_subgraph(g, kept)


# --------------------------------------------------------------------------- #
# Node dropout
# --------------------------------------------------------------------------- #

def node_dropout(
    graph: Dict[str, torch.Tensor],
    drop_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Randomly remove nodes (and their incident edges) from the graph."""
    g = _clone_graph(graph)
    x = g["x"]
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return g

    keep_mask = torch.rand(num_nodes) >= drop_ratio
    # Always keep at least one node
    if not keep_mask.any():
        keep_mask[0] = True

    kept = torch.where(keep_mask)[0].tolist()
    return _reindex_subgraph(g, kept)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _reindex_subgraph(
    g: Dict[str, torch.Tensor],
    kept_nodes: list,
) -> Dict[str, torch.Tensor]:
    """Keep only ``kept_nodes`` and reindex edges accordingly."""
    x = g["x"]
    edge_index = g["edge_index"]

    kept_set = set(kept_nodes)
    old_to_new = {old: new for new, old in enumerate(kept_nodes)}

    # Filter node features
    g["x"] = x[kept_nodes]

    # Filter and reindex edges
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        new_src, new_dst = [], []
        edge_keep_idx = []
        for i, (s, d) in enumerate(zip(src, dst)):
            if s in kept_set and d in kept_set:
                new_src.append(old_to_new[s])
                new_dst.append(old_to_new[d])
                edge_keep_idx.append(i)
        if new_src:
            g["edge_index"] = torch.tensor(
                [new_src, new_dst], dtype=torch.long
            )
            if "edge_attr" in g and g["edge_attr"] is not None:
                g["edge_attr"] = g["edge_attr"][edge_keep_idx]
        else:
            g["edge_index"] = torch.zeros(2, 0, dtype=torch.long)
            if "edge_attr" in g and g["edge_attr"] is not None:
                g["edge_attr"] = g["edge_attr"][:0]
    return g


# --------------------------------------------------------------------------- #
# Composite augmentation (pick two random augmentations)
# --------------------------------------------------------------------------- #

_AUGMENTATIONS = [
    node_feature_masking,
    edge_dropout,
    subgraph_sampling,
    node_dropout,
]


def random_augmentation(
    graph: Dict[str, torch.Tensor],
    *,
    mask_ratio: float = 0.15,
    edge_drop_ratio: float = 0.1,
    subgraph_ratio: float = 0.8,
    node_drop_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Apply a randomly chosen augmentation with default parameters."""
    idx = int(torch.randint(0, len(_AUGMENTATIONS), (1,)).item())
    aug_fn = _AUGMENTATIONS[idx]
    if aug_fn is node_feature_masking:
        return aug_fn(graph, mask_ratio=mask_ratio)
    elif aug_fn is edge_dropout:
        return aug_fn(graph, drop_ratio=edge_drop_ratio)
    elif aug_fn is subgraph_sampling:
        return aug_fn(graph, ratio=subgraph_ratio)
    else:
        return aug_fn(graph, drop_ratio=node_drop_ratio)
