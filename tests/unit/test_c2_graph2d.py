"""Phase-A C2 discriminator: graph2d/main wired through the activation gateway.

Runnable in isolation (the repo-global conftest fails to collect under the
local Python here)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_graph2d.py -q

Proves ``Graph2DClassifier._load_model`` (src/ml/vision_2d.py) never raw-loads
a checkpoint from ``self.model_path`` any more:
  * pin-absent (unconfigured store)      -> degrades, ``model is None``.
  * fixture pin (real store + manifest)  -> success, model loaded from the
    EXACT pinned bytes (verified via a distinguishing bias value baked into
    the checkpoint's state dict).
  * digest-tamper (bytes on disk changed after the manifest pins the
    original digest) -> degrades, never loads the tampered bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import List

import pytest

torch = pytest.importorskip("torch")

from src.core.model_activation import activation_gateway as gw  # noqa: E402

_LID = "graph2d/main"
_AID = "main"
_RELPATH = "graph2d_main.pth"


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


def _single_file_entry(digest: str) -> dict:
    return {
        "logical_activation_id": _LID,
        "artifact_id": _AID,
        "kind": "single_file",
        "digest": digest,
        "store_relpath": _RELPATH,
    }


def _make_checkpoint_bytes(bias_value: float) -> bytes:
    """Build a minimal, real SimpleGraphClassifier checkpoint (legacy 'gcn' branch).

    ``bias_value`` is baked into the classifier head's bias so the test can
    prove the loaded model really came from THESE bytes (not some other
    checkpoint / stale cache).
    """
    from src.ml.train.model_2d import SimpleGraphClassifier

    node_dim = 4
    hidden_dim = 8
    num_classes = 3
    model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)
    with torch.no_grad():
        model.classifier.bias.fill_(bias_value)

    checkpoint = {
        "label_map": {"a": 0, "b": 1, "c": 2},
        "node_dim": node_dim,
        "edge_dim": 0,
        "hidden_dim": hidden_dim,
        "model_type": "gcn",
        "model_state_dict": model.state_dict(),
    }
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


def _configure_pinned_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes) -> Path:
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


def _fresh_classifier_class():
    """Return the Graph2DClassifier class.

    Note: we deliberately do NOT ``importlib.reload(vision_2d)`` here. Reloading
    re-executes the module and resets its module-level singletons (``_graph2d`` /
    ``_load_error``), which the model-readiness registry reads via
    ``sys.modules.get("src.ml.vision_2d")`` — leaking a dirty module state into
    later tests (alphabetical CI order runs this file before
    test_model_readiness_registry.py). torch is imported at module top
    (``pytest.importorskip``), so ``HAS_TORCH`` is already current; a plain import
    is sufficient and side-effect-free.
    """
    import src.ml.vision_2d as vision_2d_mod

    return vision_2d_mod.Graph2DClassifier


def test_pin_absent_degrades_no_raw_load(tmp_path: Path):
    """Unconfigured store (no pin at all) -> family degrades, no raw load."""
    Graph2DClassifier = _fresh_classifier_class()

    # A real-looking checkpoint file sitting at the configured model_path,
    # to prove it is NEVER read when the gateway is unconfigured.
    bogus_path = tmp_path / "graph2d_local.pth"
    bogus_path.write_bytes(_make_checkpoint_bytes(bias_value=999.0))

    clf = Graph2DClassifier(model_path=str(bogus_path))

    assert clf.model is None
    assert clf._loaded is False
    # A missing/unverified artifact is a graceful FALLBACK, not a load error:
    # _load_error stays None so the readiness registry reports "graph2d:fallback"
    # (not "graph2d:error"), matching the pre-wiring missing-model contract.
    #
    # F4 corrected contract: readiness for this gateway-wired family is judged on
    # ACTIVATION, not legacy checkpoint-file presence. Because this un-activated
    # ``_loaded is False`` state is what the registry reads, graph2d is reported
    # as explicit "fallback" (degraded) — it can NEVER be "available" on the
    # strength of a checkpoint file merely existing on disk.
    assert clf._load_error is None


def test_fixture_pin_success_uses_exact_pinned_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A correctly pinned checkpoint -> success path loads from those exact bytes."""
    Graph2DClassifier = _fresh_classifier_class()

    data = _make_checkpoint_bytes(bias_value=42.0)
    _configure_pinned_file(monkeypatch, tmp_path, data)

    # model_path points somewhere that does NOT contain these bytes, proving
    # the load came from the activation gateway, not the filesystem path.
    missing_path = tmp_path / "does_not_exist_on_disk.pth"
    clf = Graph2DClassifier(model_path=str(missing_path))

    assert clf._load_error is None
    assert clf._loaded is True
    assert clf.model is not None
    assert torch.allclose(clf.model.classifier.bias, torch.full((3,), 42.0))


def test_digest_tamper_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bytes on disk changed after the manifest pinned the original digest -> degrade."""
    Graph2DClassifier = _fresh_classifier_class()

    original = _make_checkpoint_bytes(bias_value=7.0)
    store_root = _configure_pinned_file(monkeypatch, tmp_path, original)

    tampered = _make_checkpoint_bytes(bias_value=13.0)
    assert tampered != original
    (store_root / _RELPATH).write_bytes(tampered)

    clf = Graph2DClassifier(model_path=str(tmp_path / "irrelevant.pth"))

    assert clf.model is None
    assert clf._loaded is False
    # Digest mismatch degrades gracefully (fallback), never loads the tampered
    # bytes; _load_error stays None (readiness: "graph2d:fallback").
    assert clf._load_error is None
