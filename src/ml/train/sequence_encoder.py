"""Sequence encoder and classifier for CAD command histories.

The implementation is intentionally lightweight:
- embedding layer over discrete command ids
- GRU encoder over temporal command order
- masked mean/max pooling for stable sequence embeddings

This mirrors the sequence-centric data flow used by SketchGraphs/DeepCAD style
pipelines without importing their heavier training stacks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def _masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = mask.unsqueeze(-1)
    masked = x.masked_fill(~mask_expanded, float("-inf"))
    pooled = masked.max(dim=1).values
    pooled[~torch.isfinite(pooled)] = 0.0
    return pooled


class SequenceCommandEncoder(nn.Module):
    """Encode integer command sequences into fixed-width embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        padding_idx: int = 0,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = max(1, int(vocab_size))
        self.embedding_dim = max(1, int(embedding_dim))
        self.hidden_dim = max(1, int(hidden_dim))
        self.num_layers = max(1, int(num_layers))
        self.bidirectional = bool(bidirectional)
        self.padding_idx = max(0, int(padding_idx))
        self.dropout_rate = max(0.0, float(dropout))
        self.num_directions = 2 if self.bidirectional else 1
        self.output_dim = max(
            1,
            int(projection_dim or (self.hidden_dim * self.num_directions)),
        )

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_dim,
            padding_idx=self.padding_idx,
        )
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.projection = nn.Linear(
            self.hidden_dim * self.num_directions * 2,
            self.output_dim,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"expected [batch, seq], got shape={tuple(inputs.shape)}")

        batch_size, seq_len = inputs.shape
        if lengths is None:
            lengths = (inputs != self.padding_idx).sum(dim=1)
        lengths = lengths.to(device=inputs.device, dtype=torch.long).clamp_min(1)

        embedded = self.input_dropout(self.embedding(inputs))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_len,
        )

        steps = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)
        mean_pooled = _masked_mean_pool(encoded, mask)
        max_pooled = _masked_max_pool(encoded, mask)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        return self.output_norm(self.projection(pooled))


class SequenceCommandClassifier(nn.Module):
    """Classify CAD command histories."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = SequenceCommandEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx,
        )
        self.classifier = nn.Linear(self.encoder.output_dim, max(1, int(num_classes)))

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self.encoder(inputs, lengths=lengths)
        return self.classifier(embedding)
