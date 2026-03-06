from __future__ import annotations

import torch

from src.ml.train.sequence_encoder import (
    SequenceCommandClassifier,
    SequenceCommandEncoder,
)


def test_sequence_encoder_output_shape_and_padding_inference() -> None:
    model = SequenceCommandEncoder(
        vocab_size=32,
        embedding_dim=8,
        hidden_dim=12,
        dropout=0.0,
        padding_idx=0,
    )
    model.eval()
    inputs = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=torch.long,
    )
    explicit_lengths = torch.tensor([3, 2], dtype=torch.long)

    out_explicit = model(inputs, lengths=explicit_lengths)
    out_inferred = model(inputs)

    assert out_explicit.shape == (2, 12)
    assert torch.allclose(out_explicit, out_inferred)


def test_sequence_command_classifier_output_shape() -> None:
    model = SequenceCommandClassifier(
        vocab_size=32,
        num_classes=4,
        embedding_dim=8,
        hidden_dim=10,
        dropout=0.0,
        padding_idx=0,
    )
    model.eval()
    inputs = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)

    logits = model(inputs, lengths=lengths)

    assert logits.shape == (2, 4)

