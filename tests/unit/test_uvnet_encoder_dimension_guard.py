import os
import tempfile

import pytest

from src.ml.vision_3d import HAS_TORCH, UVNetEncoder

if HAS_TORCH:
    import torch
    from src.ml.train.model import UVNetGraphModel


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_encoder_dimension_mismatch_returns_zeros() -> None:
    model = UVNetGraphModel(
        node_input_dim=3,
        hidden_dim=4,
        embedding_dim=8,
        num_classes=2,
        dropout_rate=0.0,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.get_config(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "uvnet_test.pth")
        torch.save(checkpoint, checkpoint_path)

        encoder = UVNetEncoder(model_path=checkpoint_path)
        assert encoder._loaded is True

        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        embedding = encoder.encode({"x": x, "edge_index": edge_index})

        assert len(embedding) == model.embedding_dim
        assert all(value == 0.0 for value in embedding)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_encoder_schema_mismatch_returns_zeros() -> None:
    model = UVNetGraphModel(
        node_input_dim=3,
        hidden_dim=4,
        embedding_dim=8,
        num_classes=2,
        dropout_rate=0.0,
        node_schema=("a", "b", "c"),
        edge_schema=("e",),
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.get_config(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "uvnet_schema_test.pth")
        torch.save(checkpoint, checkpoint_path)

        encoder = UVNetEncoder(model_path=checkpoint_path)
        assert encoder._loaded is True

        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        embedding = encoder.encode(
            {
                "x": x,
                "edge_index": edge_index,
                "node_schema": ("a", "b"),
                "edge_schema": ("e",),
            }
        )

        assert len(embedding) == model.embedding_dim
        assert all(value == 0.0 for value in embedding)
