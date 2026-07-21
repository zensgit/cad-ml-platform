"""C2 discriminator: history/sequence classifier wired to the activation gateway.

Runnable in isolation (repo-global conftest fails to collect under this
Python)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_history_sequence.py -q

Proves the C4 degraded contract as actually consumed by
``HistorySequenceClassifier._load_model``:
  * pin-absent (gateway unconfigured) -> family degrades, ``_loaded_model`` is
    False, no raw path-based load is attempted, prototype fallback answers;
  * a fixture pin (real bytes, matching digest) -> SUCCESS path, the exact
    pinned checkpoint bytes are reconstructed and used for prediction;
  * digest-tamper (bytes on disk changed after pinning) -> degrades, never
    loads the tampered bytes.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from src.core.model_activation import activation_gateway as gw

torch = pytest.importorskip("torch")

from src.ml.train.sequence_encoder import SequenceCommandClassifier  # noqa: E402

_LID = "history/sequence"
_AID = "main"
_RELPATH = "history_sequence.ckpt"


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


def _build_checkpoint_bytes(tmp_path: Path) -> bytes:
    """Build a real, tiny, valid history-sequence checkpoint on disk and
    return its serialized bytes (this is the fixture artifact; production
    never reads it back off this path -- only the gateway-verified bytes are
    used)."""
    model = SequenceCommandClassifier(
        vocab_size=16,
        num_classes=2,
        embedding_dim=8,
        hidden_dim=8,
        dropout=0.0,
        padding_idx=0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 2.5]))

    scratch = tmp_path / "scratch.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": {"alpha": 0, "beta": 1},
            "model_config": {
                "vocab_size": 16,
                "embedding_dim": 8,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "padding_idx": 0,
                "max_sequence_length": 16,
            },
        },
        scratch,
    )
    return scratch.read_bytes()


def _write_manifest(path: Path, digest: str) -> Path:
    entries = [
        {
            "logical_activation_id": _LID,
            "artifact_id": _AID,
            "kind": "single_file",
            "digest": digest,
            "store_relpath": _RELPATH,
        }
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _configure_pinned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes
) -> Path:
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / _RELPATH).write_bytes(data)
    manifest = _write_manifest(tmp_path / "baseline.json", _sha256_hex(data))
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


def _fresh_classifier_module():
    """Reimport the module under test so its module-level HAS_TORCH/import
    state is unaffected, while giving us a fresh class each call (the class
    itself holds no gateway state -- state lives in the process-wide gateway
    singleton, reset via ``reset_gateway_for_tests``)."""
    import src.ml.history_sequence_classifier as mod

    return importlib.reload(mod)


def test_pin_absent_degrades_no_raw_load(tmp_path: Path) -> None:
    """Gateway unconfigured (no store root) -> family degrades even though a
    perfectly valid checkpoint sits at ``model_path`` on disk.

    This is the discriminating half of the assertion: against the OLD
    (un-wired) implementation, a valid ``model_path`` on disk was loaded
    directly via ``torch.load(model_path)`` and this test would fail (model
    would load, status would be "ok"/source "history_sequence_model"). Against
    the wired implementation the raw-load-by-path fallback is gone, so an
    unconfigured gateway degrades regardless of what sits on disk at
    ``model_path``.
    """
    data = _build_checkpoint_bytes(tmp_path)
    on_disk_checkpoint = tmp_path / "valid_but_unpinned.ckpt"
    on_disk_checkpoint.write_bytes(data)

    mod = _fresh_classifier_module()

    classifier = mod.HistorySequenceClassifier(
        prototypes_path=str(tmp_path / "missing_prototypes.json"),
        model_path=str(on_disk_checkpoint),
        min_sequence_length=2,
    )

    assert classifier.model is None
    assert classifier._loaded_model is False
    assert classifier.label_map == {}

    result = classifier.predict_from_tokens([1, 2, 3])
    assert result["source"] != "history_sequence_model"


def test_fixture_pin_success_uses_exact_pinned_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured store + matching digest -> SUCCESS: the classifier loads
    the gateway-verified checkpoint bytes and predicts via the model path."""
    data = _build_checkpoint_bytes(tmp_path)
    _configure_pinned_file(monkeypatch, tmp_path, data)

    mod = _fresh_classifier_module()
    classifier = mod.HistorySequenceClassifier(
        prototypes_path=str(tmp_path / "missing_prototypes.json"),
        model_path="",
        min_sequence_length=2,
    )

    assert classifier._loaded_model is True
    assert classifier.model is not None
    assert classifier.label_map == {"alpha": 0, "beta": 1}

    result = classifier.predict_from_tokens([1, 2, 3])
    assert result["status"] == "ok"
    assert result["source"] == "history_sequence_model"
    assert result["label"] == "beta"
    assert result["label_map_size"] == 2


def test_digest_tamper_degrades_never_loads_tampered_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bytes on disk changed after pinning (digest mismatch) -> gateway
    refuses -> family degrades, never loads the tampered artifact -- even
    though ``model_path`` also points at a separate, perfectly valid
    checkpoint on disk (proving the tamper-refusal isn't papered over by a
    raw-load-by-path fallback)."""
    data = _build_checkpoint_bytes(tmp_path)
    store_root = _configure_pinned_file(monkeypatch, tmp_path, data)

    # Tamper the on-disk artifact after the pin (and gateway) were configured
    # against the original digest.
    (store_root / _RELPATH).write_bytes(data + b"\x00tampered")

    # A valid checkpoint also happens to sit at model_path -- the OLD
    # implementation would happily torch.load() it directly and succeed; the
    # wired implementation must ignore it entirely and only ever consult the
    # gateway, which refuses on digest mismatch.
    on_disk_checkpoint = tmp_path / "valid_but_irrelevant.ckpt"
    on_disk_checkpoint.write_bytes(data)

    mod = _fresh_classifier_module()
    classifier = mod.HistorySequenceClassifier(
        prototypes_path=str(tmp_path / "missing_prototypes.json"),
        model_path=str(on_disk_checkpoint),
        min_sequence_length=2,
    )

    assert classifier.model is None
    assert classifier._loaded_model is False

    result = classifier.predict_from_tokens([1, 2, 3])
    assert result["source"] != "history_sequence_model"
