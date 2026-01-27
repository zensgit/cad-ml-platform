from __future__ import annotations

import pytest


def test_class_balancer_weights() -> None:
    torch = pytest.importorskip("torch")
    from src.ml.class_balancer import ClassBalancer

    labels = [0, 0, 0, 1]
    balancer = ClassBalancer(strategy="weights", weight_mode="inverse")
    weights = balancer.compute_class_weights(labels, num_classes=2)

    assert weights.shape[0] == 2
    # Minority class (1) should receive higher weight than class 0
    assert weights[1] > weights[0]


def test_class_balancer_focal_loss_type() -> None:
    pytest.importorskip("torch")
    from src.ml.class_balancer import ClassBalancer, FocalLoss

    labels = [0, 0, 1, 1]
    balancer = ClassBalancer(strategy="focal", weight_mode="sqrt", focal_alpha=0.5, focal_gamma=2.0)
    loss_fn = balancer.get_loss_function(labels=labels, num_classes=2)

    assert isinstance(loss_fn, FocalLoss)


def test_class_balancer_logit_adjusted_type() -> None:
    pytest.importorskip("torch")
    from src.ml.class_balancer import ClassBalancer, LogitAdjustedLoss

    labels = [0, 0, 1, 1]
    balancer = ClassBalancer(strategy="logit_adj", logit_adj_tau=1.0)
    loss_fn = balancer.get_loss_function(labels=labels, num_classes=2)

    assert isinstance(loss_fn, LogitAdjustedLoss)
