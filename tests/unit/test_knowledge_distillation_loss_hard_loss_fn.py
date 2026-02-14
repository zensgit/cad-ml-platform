from __future__ import annotations

import torch
from torch import nn

from src.ml.knowledge_distillation import DistillationLoss


def test_distillation_loss_uses_custom_hard_loss_fn(monkeypatch) -> None:
    monkeypatch.delenv("DISTILLATION_ALPHA", raising=False)
    monkeypatch.delenv("DISTILLATION_TEMPERATURE", raising=False)

    # Two samples, two classes.
    student_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    teacher_logits = torch.zeros_like(student_logits)
    hard_labels = torch.tensor([0, 1], dtype=torch.long)

    # Make class 0 "more expensive" to ensure weighting changes the result.
    weights = torch.tensor([2.0, 1.0], dtype=torch.float)
    hard_loss_fn = nn.CrossEntropyLoss(weight=weights)

    loss_fn = DistillationLoss(alpha=1.0, temperature=1.0, hard_loss_fn=hard_loss_fn)
    loss, components = loss_fn(student_logits, teacher_logits, hard_labels)

    expected = hard_loss_fn(student_logits, hard_labels)
    assert torch.isclose(loss, expected)
    assert abs(components["ce_loss"] - expected.item()) < 1e-8

