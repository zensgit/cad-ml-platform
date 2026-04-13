"""Tests for src/ml/graph_augmentations.py."""

from __future__ import annotations

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")

if TORCH_AVAILABLE:
    from src.ml.graph_augmentations import (
        edge_dropout,
        node_dropout,
        node_feature_masking,
        random_augmentation,
        subgraph_sampling,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_graph(num_nodes: int = 10, node_dim: int = 19, edge_dim: int = 7):
    """Create a simple test graph."""
    x = torch.randn(num_nodes, node_dim)
    # Chain edges: 0->1->2->...->n-1
    if num_nodes > 1:
        src = list(range(num_nodes - 1))
        dst = list(range(1, num_nodes))
        # Add reverse edges for undirected
        src_all = src + dst
        dst_all = dst + src
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def _empty_graph(node_dim: int = 19, edge_dim: int = 7):
    return {
        "x": torch.zeros(0, node_dim),
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "edge_attr": torch.zeros(0, edge_dim),
    }


def _single_node_graph(node_dim: int = 19, edge_dim: int = 7):
    return {
        "x": torch.randn(1, node_dim),
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "edge_attr": torch.zeros(0, edge_dim),
    }


# --------------------------------------------------------------------------- #
# node_feature_masking
# --------------------------------------------------------------------------- #

class TestNodeFeatureMasking:
    def test_does_not_modify_original(self):
        g = _make_graph()
        original_x = g["x"].clone()
        node_feature_masking(g, mask_ratio=0.5)
        assert torch.equal(g["x"], original_x)

    def test_shape_preserved(self):
        g = _make_graph(num_nodes=20, node_dim=19)
        result = node_feature_masking(g, mask_ratio=0.3)
        assert result["x"].shape == g["x"].shape

    def test_some_features_zeroed(self):
        g = _make_graph(num_nodes=50, node_dim=19)
        torch.manual_seed(0)
        result = node_feature_masking(g, mask_ratio=0.5)
        # With 50*19=950 features and 50% masking, we expect ~475 zeros
        zeros_count = (result["x"] == 0).sum().item()
        assert zeros_count > 0

    def test_mask_ratio_zero(self):
        g = _make_graph()
        result = node_feature_masking(g, mask_ratio=0.0)
        assert torch.equal(result["x"], g["x"])

    def test_edges_unchanged(self):
        g = _make_graph()
        result = node_feature_masking(g, mask_ratio=0.3)
        assert torch.equal(result["edge_index"], g["edge_index"])

    def test_empty_graph(self):
        g = _empty_graph()
        result = node_feature_masking(g, mask_ratio=0.5)
        assert result["x"].shape == (0, 19)

    def test_single_node(self):
        g = _single_node_graph()
        result = node_feature_masking(g, mask_ratio=0.5)
        assert result["x"].shape == (1, 19)


# --------------------------------------------------------------------------- #
# edge_dropout
# --------------------------------------------------------------------------- #

class TestEdgeDropout:
    def test_does_not_modify_original(self):
        g = _make_graph()
        original_edges = g["edge_index"].clone()
        edge_dropout(g, drop_ratio=0.5)
        assert torch.equal(g["edge_index"], original_edges)

    def test_fewer_edges(self):
        g = _make_graph(num_nodes=20)
        torch.manual_seed(42)
        result = edge_dropout(g, drop_ratio=0.5)
        assert result["edge_index"].size(1) <= g["edge_index"].size(1)

    def test_edge_attr_consistent(self):
        g = _make_graph(num_nodes=20)
        result = edge_dropout(g, drop_ratio=0.3)
        assert result["edge_attr"].size(0) == result["edge_index"].size(1)

    def test_drop_ratio_zero(self):
        g = _make_graph()
        result = edge_dropout(g, drop_ratio=0.0)
        assert result["edge_index"].size(1) == g["edge_index"].size(1)

    def test_empty_graph(self):
        g = _empty_graph()
        result = edge_dropout(g, drop_ratio=0.5)
        assert result["edge_index"].size(1) == 0

    def test_node_features_unchanged(self):
        g = _make_graph()
        result = edge_dropout(g, drop_ratio=0.3)
        assert torch.equal(result["x"], g["x"])


# --------------------------------------------------------------------------- #
# subgraph_sampling
# --------------------------------------------------------------------------- #

class TestSubgraphSampling:
    def test_does_not_modify_original(self):
        g = _make_graph()
        original_x = g["x"].clone()
        subgraph_sampling(g, ratio=0.5)
        assert torch.equal(g["x"], original_x)

    def test_fewer_nodes(self):
        g = _make_graph(num_nodes=20)
        torch.manual_seed(0)
        result = subgraph_sampling(g, ratio=0.5)
        assert result["x"].size(0) <= g["x"].size(0)
        assert result["x"].size(0) >= 1

    def test_edges_reindexed(self):
        g = _make_graph(num_nodes=20)
        result = subgraph_sampling(g, ratio=0.5)
        num_nodes = result["x"].size(0)
        if result["edge_index"].numel() > 0:
            assert result["edge_index"].max().item() < num_nodes

    def test_ratio_one(self):
        g = _make_graph(num_nodes=10)
        result = subgraph_sampling(g, ratio=1.0)
        # Should keep all nodes (BFS from random start in connected graph)
        assert result["x"].size(0) == g["x"].size(0)

    def test_edge_attr_consistent(self):
        g = _make_graph(num_nodes=20)
        result = subgraph_sampling(g, ratio=0.5)
        assert result["edge_attr"].size(0) == result["edge_index"].size(1)

    def test_empty_graph(self):
        g = _empty_graph()
        result = subgraph_sampling(g, ratio=0.5)
        assert result["x"].size(0) == 0

    def test_single_node(self):
        g = _single_node_graph()
        result = subgraph_sampling(g, ratio=0.5)
        assert result["x"].size(0) == 1


# --------------------------------------------------------------------------- #
# node_dropout
# --------------------------------------------------------------------------- #

class TestNodeDropout:
    def test_does_not_modify_original(self):
        g = _make_graph()
        original_x = g["x"].clone()
        node_dropout(g, drop_ratio=0.5)
        assert torch.equal(g["x"], original_x)

    def test_fewer_nodes(self):
        g = _make_graph(num_nodes=50)
        torch.manual_seed(0)
        result = node_dropout(g, drop_ratio=0.5)
        assert result["x"].size(0) <= g["x"].size(0)
        assert result["x"].size(0) >= 1  # Always keeps at least 1

    def test_edges_reindexed(self):
        g = _make_graph(num_nodes=20)
        result = node_dropout(g, drop_ratio=0.3)
        num_nodes = result["x"].size(0)
        if result["edge_index"].numel() > 0:
            assert result["edge_index"].max().item() < num_nodes

    def test_edge_attr_consistent(self):
        g = _make_graph(num_nodes=20)
        result = node_dropout(g, drop_ratio=0.3)
        assert result["edge_attr"].size(0) == result["edge_index"].size(1)

    def test_drop_ratio_zero(self):
        g = _make_graph()
        result = node_dropout(g, drop_ratio=0.0)
        assert result["x"].size(0) == g["x"].size(0)

    def test_empty_graph(self):
        g = _empty_graph()
        result = node_dropout(g, drop_ratio=0.5)
        assert result["x"].size(0) == 0

    def test_single_node(self):
        g = _single_node_graph()
        result = node_dropout(g, drop_ratio=0.9)
        assert result["x"].size(0) == 1  # Cannot drop the only node


# --------------------------------------------------------------------------- #
# random_augmentation
# --------------------------------------------------------------------------- #

class TestRandomAugmentation:
    def test_returns_valid_graph(self):
        g = _make_graph()
        result = random_augmentation(g)
        assert "x" in result
        assert "edge_index" in result
        assert result["x"].dim() == 2
        assert result["edge_index"].dim() == 2

    def test_does_not_modify_original(self):
        g = _make_graph()
        original_x = g["x"].clone()
        random_augmentation(g)
        assert torch.equal(g["x"], original_x)
