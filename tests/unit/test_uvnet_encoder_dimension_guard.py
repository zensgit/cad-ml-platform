import hashlib
import json
import os
import tempfile

import pytest

from src.ml.vision_3d import HAS_TORCH, UVNetEncoder

if HAS_TORCH:
    import torch
    from src.ml.train.model import UVNetGraphModel

# Phase-A decision #3: UVNetEncoder no longer raw-loads ``self.model_path`` —
# bytes come only from the C1 activation gateway (pin + sha-256 verified). These
# guards still test the dimension/schema mismatch -> zeros contract, but the
# encoder must first be ACTIVATED by pinning the checkpoint bytes through the
# gateway (a raw path is never read any more), otherwise ``_loaded`` is False and
# the guard would never be reached.
_LID = "vision3d-uvnet/main"
_AID = "main"
_RELPATH = "uvnet_main.pth"


@pytest.fixture(autouse=True)
def _clean_gateway(monkeypatch):
    if not HAS_TORCH:
        yield
        return
    from src.core.model_activation import activation_gateway as gw

    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    yield
    gw.reset_gateway_for_tests()


def _pin_checkpoint(monkeypatch, tmpdir: str, data: bytes) -> None:
    """Pin ``data`` as the vision3d-uvnet/main SINGLE_FILE activation."""
    from src.core.model_activation import activation_gateway as gw

    store_root = os.path.join(tmpdir, "store")
    os.makedirs(store_root, exist_ok=True)
    with open(os.path.join(store_root, _RELPATH), "wb") as f:
        f.write(data)
    manifest = os.path.join(tmpdir, "baseline.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "logical_activation_id": _LID,
                    "artifact_id": _AID,
                    "kind": "single_file",
                    "digest": hashlib.sha256(data).hexdigest(),
                    "store_relpath": _RELPATH,
                }
            ],
            f,
        )
    monkeypatch.setenv(gw.ENV_STORE_ROOT, store_root)
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", manifest)
    gw.reset_gateway_for_tests()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_encoder_dimension_mismatch_returns_zeros(monkeypatch) -> None:
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
        buf = os.path.join(tmpdir, "uvnet_test.pth")
        torch.save(checkpoint, buf)
        with open(buf, "rb") as f:
            data = f.read()
        # Activate the encoder via the gateway (raw model_path is never read).
        _pin_checkpoint(monkeypatch, tmpdir, data)

        encoder = UVNetEncoder(model_path=os.path.join(tmpdir, "unused.pth"))
        assert encoder._loaded is True

        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        embedding = encoder.encode({"x": x, "edge_index": edge_index})

        assert len(embedding) == model.embedding_dim
        assert all(value == 0.0 for value in embedding)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_encoder_schema_mismatch_returns_zeros(monkeypatch) -> None:
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
        buf = os.path.join(tmpdir, "uvnet_schema_test.pth")
        torch.save(checkpoint, buf)
        with open(buf, "rb") as f:
            data = f.read()
        _pin_checkpoint(monkeypatch, tmpdir, data)

        encoder = UVNetEncoder(model_path=os.path.join(tmpdir, "unused.pth"))
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
