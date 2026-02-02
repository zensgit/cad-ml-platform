"""
Graph augmentations for GNN-based models.

Provides augmentations that modify graph structure and features.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class GraphAugmentation(ABC):
    """Base class for graph augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Initialize augmentation.

        Args:
            p: Probability of applying this augmentation
        """
        self.p = p

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation to graph data."""
        if random.random() > self.p:
            return data
        return self._apply(data)

    @abstractmethod
    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation (internal)."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get augmentation parameters."""
        return {"p": self.p}


class NodeDropout(GraphAugmentation):
    """
    Randomly drop nodes from the graph.

    Also removes associated edges.
    """

    def __init__(self, dropout_rate: float = 0.1, p: float = 0.5):
        """
        Initialize node dropout.

        Args:
            dropout_rate: Fraction of nodes to drop
            p: Probability of applying
        """
        super().__init__(p)
        self.dropout_rate = dropout_rate

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            x = data.get("x")
            edge_index = data.get("edge_index")

            if x is None or edge_index is None:
                return data

            num_nodes = x.shape[0]
            if num_nodes <= 1:
                return data

            # Determine nodes to keep
            num_drop = max(1, int(num_nodes * self.dropout_rate))
            num_keep = num_nodes - num_drop

            if num_keep < 1:
                return data

            # Random permutation to select nodes to keep
            perm = torch.randperm(num_nodes)
            keep_indices = perm[:num_keep].sort().values
            keep_set = set(keep_indices.tolist())

            # Create mapping from old to new indices
            old_to_new = {old: new for new, old in enumerate(keep_indices.tolist())}

            # Filter node features
            data["x"] = x[keep_indices]

            # Filter and remap edges
            src, dst = edge_index[0], edge_index[1]
            mask = torch.tensor([
                s.item() in keep_set and d.item() in keep_set
                for s, d in zip(src, dst)
            ])

            if mask.sum() > 0:
                new_src = torch.tensor([old_to_new[s.item()] for s, m in zip(src, mask) if m])
                new_dst = torch.tensor([old_to_new[d.item()] for d, m in zip(dst, mask) if m])
                data["edge_index"] = torch.stack([new_src, new_dst])
            else:
                # No edges remain
                data["edge_index"] = torch.zeros((2, 0), dtype=torch.long)

            # Filter edge attributes if present
            if "edge_attr" in data and data["edge_attr"] is not None:
                data["edge_attr"] = data["edge_attr"][mask]

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "dropout_rate": self.dropout_rate}


class EdgeDropout(GraphAugmentation):
    """Randomly drop edges from the graph."""

    def __init__(self, dropout_rate: float = 0.1, p: float = 0.5):
        """
        Initialize edge dropout.

        Args:
            dropout_rate: Fraction of edges to drop
            p: Probability of applying
        """
        super().__init__(p)
        self.dropout_rate = dropout_rate

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            edge_index = data.get("edge_index")
            if edge_index is None:
                return data

            num_edges = edge_index.shape[1]
            if num_edges == 0:
                return data

            # Create mask for edges to keep
            keep_prob = 1.0 - self.dropout_rate
            mask = torch.rand(num_edges) < keep_prob

            # Ensure at least one edge remains
            if mask.sum() == 0:
                mask[random.randint(0, num_edges - 1)] = True

            data["edge_index"] = edge_index[:, mask]

            # Filter edge attributes if present
            if "edge_attr" in data and data["edge_attr"] is not None:
                data["edge_attr"] = data["edge_attr"][mask]

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "dropout_rate": self.dropout_rate}


class NodeFeatureNoise(GraphAugmentation):
    """Add Gaussian noise to node features."""

    def __init__(
        self,
        noise_scale: float = 0.1,
        feature_indices: Optional[List[int]] = None,
        p: float = 0.5,
    ):
        """
        Initialize node feature noise.

        Args:
            noise_scale: Standard deviation of noise
            feature_indices: Indices of features to perturb (None for all)
            p: Probability of applying
        """
        super().__init__(p)
        self.noise_scale = noise_scale
        self.feature_indices = feature_indices

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            x = data.get("x")
            if x is None:
                return data

            # Clone to avoid modifying original
            x = x.clone()

            if self.feature_indices is None:
                # Add noise to all features
                noise = torch.randn_like(x) * self.noise_scale
                x = x + noise
            else:
                # Add noise to selected features
                for idx in self.feature_indices:
                    if idx < x.shape[1]:
                        noise = torch.randn(x.shape[0]) * self.noise_scale
                        x[:, idx] = x[:, idx] + noise

            data["x"] = x
            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "noise_scale": self.noise_scale,
            "feature_indices": self.feature_indices,
        }


class EdgeFeaturePerturbation(GraphAugmentation):
    """Perturb edge features."""

    def __init__(
        self,
        noise_scale: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize edge feature perturbation.

        Args:
            noise_scale: Standard deviation of noise
            p: Probability of applying
        """
        super().__init__(p)
        self.noise_scale = noise_scale

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            edge_attr = data.get("edge_attr")
            if edge_attr is None:
                return data

            edge_attr = edge_attr.clone()
            noise = torch.randn_like(edge_attr) * self.noise_scale
            data["edge_attr"] = edge_attr + noise

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "noise_scale": self.noise_scale}


class SubgraphSampling(GraphAugmentation):
    """
    Sample a connected subgraph.

    Useful for creating smaller training examples from large graphs.
    """

    def __init__(
        self,
        num_nodes: Optional[int] = None,
        ratio: float = 0.8,
        p: float = 0.5,
    ):
        """
        Initialize subgraph sampling.

        Args:
            num_nodes: Exact number of nodes to sample (overrides ratio)
            ratio: Fraction of nodes to sample
            p: Probability of applying
        """
        super().__init__(p)
        self.num_nodes = num_nodes
        self.ratio = ratio

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            x = data.get("x")
            edge_index = data.get("edge_index")

            if x is None or edge_index is None:
                return data

            num_nodes = x.shape[0]

            # Determine target size
            if self.num_nodes is not None:
                target_size = min(self.num_nodes, num_nodes)
            else:
                target_size = max(1, int(num_nodes * self.ratio))

            if target_size >= num_nodes:
                return data

            # Build adjacency list
            adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
            src, dst = edge_index[0], edge_index[1]
            for s, d in zip(src.tolist(), dst.tolist()):
                adj[s].add(d)
                adj[d].add(s)

            # BFS from random start node
            start_node = random.randint(0, num_nodes - 1)
            visited = {start_node}
            queue = [start_node]

            while len(visited) < target_size and queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) >= target_size:
                            break

            # If BFS didn't find enough nodes, add random nodes
            if len(visited) < target_size:
                remaining = [i for i in range(num_nodes) if i not in visited]
                random.shuffle(remaining)
                for node in remaining[:target_size - len(visited)]:
                    visited.add(node)

            # Create node mapping
            keep_indices = sorted(visited)
            old_to_new = {old: new for new, old in enumerate(keep_indices)}
            keep_set = set(keep_indices)

            # Filter nodes
            data["x"] = x[torch.tensor(keep_indices)]

            # Filter and remap edges
            mask = torch.tensor([
                s.item() in keep_set and d.item() in keep_set
                for s, d in zip(src, dst)
            ])

            if mask.sum() > 0:
                new_src = torch.tensor([old_to_new[s.item()] for s, m in zip(src, mask) if m])
                new_dst = torch.tensor([old_to_new[d.item()] for d, m in zip(dst, mask) if m])
                data["edge_index"] = torch.stack([new_src, new_dst])
            else:
                data["edge_index"] = torch.zeros((2, 0), dtype=torch.long)

            # Filter edge attributes
            if "edge_attr" in data and data["edge_attr"] is not None:
                data["edge_attr"] = data["edge_attr"][mask]

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "num_nodes": self.num_nodes, "ratio": self.ratio}


class GraphMixup(GraphAugmentation):
    """
    Mixup augmentation for graphs.

    Interpolates node features between two graphs.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 0.5,
    ):
        """
        Initialize graph mixup.

        Args:
            alpha: Beta distribution parameter for mixup ratio
            p: Probability of applying
        """
        super().__init__(p)
        self.alpha = alpha
        self._other_graph: Optional[Dict[str, Any]] = None

    def set_mixup_graph(self, other: Dict[str, Any]) -> None:
        """Set the graph to mixup with."""
        self._other_graph = other

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._other_graph is None:
            return data

        try:
            import torch

            x1 = data.get("x")
            x2 = self._other_graph.get("x")

            if x1 is None or x2 is None:
                return data

            # Sample mixup ratio
            lam = random.betavariate(self.alpha, self.alpha)

            # Pad to same size
            max_nodes = max(x1.shape[0], x2.shape[0])
            if x1.shape[0] < max_nodes:
                padding = torch.zeros(max_nodes - x1.shape[0], x1.shape[1])
                x1 = torch.cat([x1, padding], dim=0)
            if x2.shape[0] < max_nodes:
                padding = torch.zeros(max_nodes - x2.shape[0], x2.shape[1])
                x2 = torch.cat([x2, padding], dim=0)

            # Mixup
            data["x"] = lam * x1 + (1 - lam) * x2
            data["mixup_lambda"] = lam

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "alpha": self.alpha}


class EdgeAddition(GraphAugmentation):
    """Add random edges to the graph."""

    def __init__(
        self,
        num_edges: Optional[int] = None,
        ratio: float = 0.1,
        p: float = 0.5,
    ):
        """
        Initialize edge addition.

        Args:
            num_edges: Exact number of edges to add
            ratio: Fraction of existing edges to add
            p: Probability of applying
        """
        super().__init__(p)
        self.num_edges = num_edges
        self.ratio = ratio

    def _apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import torch

            x = data.get("x")
            edge_index = data.get("edge_index")

            if x is None or edge_index is None:
                return data

            num_nodes = x.shape[0]
            num_existing = edge_index.shape[1]

            # Determine number of edges to add
            if self.num_edges is not None:
                num_add = self.num_edges
            else:
                num_add = max(1, int(num_existing * self.ratio))

            # Get existing edges as set
            existing = set()
            for i in range(num_existing):
                s, d = edge_index[0, i].item(), edge_index[1, i].item()
                existing.add((s, d))

            # Add random edges
            new_edges = []
            attempts = 0
            max_attempts = num_add * 10

            while len(new_edges) < num_add and attempts < max_attempts:
                s = random.randint(0, num_nodes - 1)
                d = random.randint(0, num_nodes - 1)
                if s != d and (s, d) not in existing:
                    new_edges.append([s, d])
                    existing.add((s, d))
                attempts += 1

            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).T
                data["edge_index"] = torch.cat([edge_index, new_edge_tensor], dim=1)

                # Add zero edge attributes for new edges if edge_attr exists
                if "edge_attr" in data and data["edge_attr"] is not None:
                    edge_dim = data["edge_attr"].shape[1]
                    new_attr = torch.zeros(len(new_edges), edge_dim)
                    data["edge_attr"] = torch.cat([data["edge_attr"], new_attr], dim=0)

            return data

        except ImportError:
            return data

    def get_params(self) -> Dict[str, Any]:
        return {"p": self.p, "num_edges": self.num_edges, "ratio": self.ratio}
