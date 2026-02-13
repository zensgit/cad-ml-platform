from __future__ import annotations

import pytest


def _require_torch():
    try:
        import torch
    except Exception:  # noqa: BLE001
        pytest.skip("torch not available")
    return torch


def test_simple_graph_classifier_batch_matches_individual():
    torch = _require_torch()

    from src.ml.train.model_2d import SimpleGraphClassifier

    torch.manual_seed(0)
    node_dim = 4
    hidden_dim = 8
    num_classes = 5

    model = SimpleGraphClassifier(
        node_dim=node_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=0.0
    )
    model.eval()

    x1 = torch.randn(3, node_dim)
    edge1 = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)

    x2 = torch.randn(2, node_dim)
    edge2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    logits1 = model(x1, edge1)
    logits2 = model(x2, edge2)

    offset = int(x1.size(0))
    x_batch = torch.cat([x1, x2], dim=0)
    edge_batch = torch.cat([edge1, edge2 + offset], dim=1)
    batch_vec = torch.cat(
        [
            torch.zeros((x1.size(0),), dtype=torch.long),
            torch.ones((x2.size(0),), dtype=torch.long),
        ],
        dim=0,
    )

    logits_batch = model(x_batch, edge_batch, batch=batch_vec)
    assert tuple(logits_batch.shape) == (2, num_classes)
    assert torch.allclose(logits_batch[0], logits1[0], atol=1e-6)
    assert torch.allclose(logits_batch[1], logits2[0], atol=1e-6)


def test_edge_sage_classifier_batch_matches_individual():
    torch = _require_torch()

    from src.ml.train.model_2d import EdgeGraphSageClassifier

    torch.manual_seed(0)
    node_dim = 4
    edge_dim = 2
    hidden_dim = 8
    num_classes = 5

    model = EdgeGraphSageClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.0,
    )
    model.eval()

    x1 = torch.randn(3, node_dim)
    edge1 = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    edge_attr1 = torch.randn(edge1.size(1), edge_dim)

    x2 = torch.randn(2, node_dim)
    edge2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr2 = torch.randn(edge2.size(1), edge_dim)

    logits1 = model(x1, edge1, edge_attr1)
    logits2 = model(x2, edge2, edge_attr2)

    offset = int(x1.size(0))
    x_batch = torch.cat([x1, x2], dim=0)
    edge_batch = torch.cat([edge1, edge2 + offset], dim=1)
    edge_attr_batch = torch.cat([edge_attr1, edge_attr2], dim=0)
    batch_vec = torch.cat(
        [
            torch.zeros((x1.size(0),), dtype=torch.long),
            torch.ones((x2.size(0),), dtype=torch.long),
        ],
        dim=0,
    )

    logits_batch = model(x_batch, edge_batch, edge_attr_batch, batch=batch_vec)
    assert tuple(logits_batch.shape) == (2, num_classes)
    assert torch.allclose(logits_batch[0], logits1[0], atol=1e-6)
    assert torch.allclose(logits_batch[1], logits2[0], atol=1e-6)

