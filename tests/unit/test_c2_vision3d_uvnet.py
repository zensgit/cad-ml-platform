"""Phase-A C2 discriminator — vision3d-uvnet/main wiring.

Proves ``UVNetEncoder._load_model`` (src/ml/vision_3d.py) goes through the C4
activation gateway (``activate_file("vision3d-uvnet/main", "main")``) instead
of a raw ``torch.load(path)``, and that the family's existing degrade branch
(``_loaded=False`` / mock embedding) is the ONLY outcome on any refusal —
never a raw-load fallback.

Runnable in isolation (the repo-global conftest fails to collect under some
Python/FastAPI combinations)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_vision3d_uvnet.py -q

Deviation from the standard py3.11 run instruction, recorded honestly: this
sandbox's ``python3.11`` has no ``torch`` installed at all (``HAS_TORCH`` is
False), so the success/tamper paths — which only execute inside
``if HAS_TORCH:`` — cannot be exercised there. This box's ``/usr/bin/python3``
(3.9.6) *does* have a real torch (2.8.0) install, so this file is verified
with::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest tests/unit/test_c2_vision3d_uvnet.py -q

The pin-absent test runs (and passes) identically under either interpreter,
since it only requires the ``HAS_TORCH=False`` mock-mode branch OR a
gateway-degrade with a torch-less no-op — see the test body for the guard.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from typing import List

import pytest

from src.core.model_activation import activation_gateway as gw

_LID = "vision3d-uvnet/main"
_AID = "main"
_RELPATH = "uvnet_main.pth"


@pytest.fixture(autouse=True)
def _clean_gateway(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test: no inherited env, fresh gateway before and after."""
    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    yield
    gw.reset_gateway_for_tests()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(path: Path, entries: List[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _single_file_entry(digest: str, relpath: str = _RELPATH) -> dict:
    return {
        "logical_activation_id": _LID,
        "artifact_id": _AID,
        "kind": "single_file",
        "digest": digest,
        "store_relpath": relpath,
    }


def _configure_pinned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes
) -> Path:
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / _RELPATH).write_bytes(data)
    manifest = _write_manifest(
        tmp_path / "baseline.json", [_single_file_entry(_sha256_hex(data))]
    )
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


def _tiny_checkpoint_bytes() -> bytes:
    """Build a real, tiny UVNetGraphModel checkpoint (state_dict + config)."""
    import torch

    from src.ml.train.model import UVNetGraphModel

    node_input_dim = 4
    edge_input_dim = 2
    hidden_dim = 8
    embedding_dim = 16
    num_classes = 3

    model = UVNetGraphModel(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout_rate=0.0,
    )
    checkpoint = {
        "config": {
            "node_input_dim": node_input_dim,
            "edge_input_dim": edge_input_dim,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "dropout_rate": 0.0,
        },
        "model_state_dict": model.state_dict(),
    }
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. pin-absent -> family degrades (no raw load)
# ---------------------------------------------------------------------------

def test_pin_absent_degrades_no_raw_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No store configured (default NO-PIN) -> encoder degrades to mock mode.

    Also plants a *real* file at the family's default env-path location to
    prove the family does NOT fall back to a raw path-based load when the
    gateway degrades.
    """
    from src.ml import vision_3d as v

    importlib.reload(v)

    decoy = tmp_path / "decoy_uvnet.pth"
    decoy.write_bytes(b"not a real checkpoint, must never be loaded")
    monkeypatch.setenv("UVNET_MODEL_PATH", str(decoy))

    encoder = v.UVNetEncoder()

    assert encoder._loaded is False
    assert encoder.model is None
    # Confirms the degrade path never attempted torch.load against the decoy
    # file: no load-error would be recorded from a garbage-bytes raw load.
    if v.HAS_TORCH:
        assert encoder._load_error is None


# ---------------------------------------------------------------------------
# 2. fixture pin -> success path uses the exact gateway-verified bytes
# ---------------------------------------------------------------------------

def test_fixture_pin_activates_and_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.ml import vision_3d as v

    importlib.reload(v)

    if not v.HAS_TORCH:
        pytest.skip(
            "torch not installed under this interpreter; HAS_TORCH=False means "
            "_load_model short-circuits before calling activate_file at all. "
            "Verify with /usr/bin/python3 (has torch) instead — see module docstring."
        )

    data = _tiny_checkpoint_bytes()
    _configure_pinned_file(monkeypatch, tmp_path, data)

    # Also plant a decoy at the legacy env-path to prove it is never touched.
    decoy = tmp_path / "decoy_uvnet.pth"
    decoy.write_bytes(b"not a real checkpoint")
    monkeypatch.setenv("UVNET_MODEL_PATH", str(decoy))

    encoder = v.UVNetEncoder()

    assert encoder._loaded is True
    assert encoder._load_error is None
    assert encoder.model is not None
    assert encoder.model.embedding_dim == 16


# ---------------------------------------------------------------------------
# 3. digest-tamper -> degrades (never loads tampered bytes)
# ---------------------------------------------------------------------------

def test_digest_tamper_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from src.ml import vision_3d as v

    importlib.reload(v)

    if not v.HAS_TORCH:
        pytest.skip(
            "torch not installed under this interpreter; see module docstring."
        )

    data = _tiny_checkpoint_bytes()
    store_root = _configure_pinned_file(monkeypatch, tmp_path, data)

    # Tamper the on-disk artifact AFTER the manifest pinned its original digest.
    (store_root / _RELPATH).write_bytes(data + b"\x00tampered")

    encoder = v.UVNetEncoder()

    assert encoder._loaded is False
    assert encoder.model is None
