"""Phase-A C2 discriminators for the pickle-classifier family wiring.

The classifier's ``load_model`` was rewired to activate its weights through the
controlled-store gateway (``activate_file("pickle-classifier/main", "main")``)
instead of raw-loading a pickle by filesystem path. These tests prove the
ratified fail-closed posture at the FAMILY boundary (not just the gateway):

  * pin-absent (store unconfigured) WHILE a real, valid, loadable pickle sits at
    ``CLASSIFICATION_MODEL_PATH`` on disk → the family degrades to
    ``model_unavailable``; it does NOT raw-load that file. This is the load-
    bearing proof that the old path fallback is gone.
  * a fixture pin (real store + baseline manifest) → SUCCESS: the exact pinned
    bytes flow through (``_MODEL_HASH`` == sha256(bytes)[:16]) and ``predict``
    returns the pickled model's label, not ``model_unavailable``.
  * digest-tamper (file swapped after pinning) → the family degrades to
    ``model_unavailable`` (never the tampered bytes).

Run in isolation (the repo-global conftest fails to collect under the local
Python)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_pickle_classifier.py -q
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import List

import pytest

from src.core.model_activation import activation_gateway as gw
from src.ml import classifier


# A top-level, importable model so ``pickle.loads`` can resolve it in-process.
class FakeModel:
    """Minimal sklearn-like model: predict returns a constant label."""

    LABEL = "fake_label"

    def predict(self, rows: List[List[float]]) -> List[str]:
        return [self.LABEL for _ in rows]


_LID = "pickle-classifier/main"
_AID = "main"
_RELPATH = "model.pkl"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _single_file_entry(digest: str, relpath: str = _RELPATH) -> dict:
    return {
        "logical_activation_id": _LID,
        "artifact_id": _AID,
        "kind": "single_file",
        "digest": digest,
        "store_relpath": relpath,
    }


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    """Reset gateway + classifier module globals before and after each test.

    ``load_model`` caches ``_MODEL`` in a module global and short-circuits on a
    fast path keyed on version/path, so a leaked ``_MODEL`` would make the
    success/degrade assertions read stale state.
    """
    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    monkeypatch.delenv("CLASSIFICATION_MODEL_PATH", raising=False)
    monkeypatch.delenv("CLASSIFICATION_MODEL_VERSION", raising=False)

    monkeypatch.setattr(classifier, "_MODEL", None, raising=False)
    monkeypatch.setattr(classifier, "_MODEL_HASH", None, raising=False)
    monkeypatch.setattr(classifier, "_MODEL_VERSION", "none", raising=False)
    monkeypatch.setattr(
        classifier, "_MODEL_PATH", Path("models/classifier_v1.pkl"), raising=False
    )
    gw.reset_gateway_for_tests()
    yield
    gw.reset_gateway_for_tests()


def _configure_pinned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes
) -> Path:
    """Create a store_root + valid manifest pinning ``data``; wire env; reset."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / _RELPATH).write_bytes(data)
    manifest = tmp_path / "baseline.json"
    manifest.write_text(
        json.dumps([_single_file_entry(_sha256_hex(data))]), encoding="utf-8"
    )
    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


# ---------------------------------------------------------------------------
# pin-absent → degrade, NO raw-load-by-path (the load-bearing discriminator)
# ---------------------------------------------------------------------------

def test_pin_absent_degrades_and_does_not_raw_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real valid pickle on disk MUST NOT load when the store is unconfigured.

    Old code did ``_MODEL_PATH.open(...); pickle.load(f)`` and would have set
    ``_MODEL`` from this file. The wired code has no store pin, so
    ``activate_file`` returns None and the family degrades. This is the proof
    the raw path fallback was removed.
    """
    on_disk = tmp_path / "real_classifier.pkl"
    on_disk.write_bytes(pickle.dumps(FakeModel()))
    assert on_disk.exists()  # a genuinely loadable pickle sits at the path
    monkeypatch.setenv("CLASSIFICATION_MODEL_PATH", str(on_disk))
    # No MODEL_ACTIVATION_STORE_ROOT → gateway unconfigured → degrade.

    classifier.load_model()
    assert classifier._MODEL is None  # did NOT raw-load the on-disk pickle
    assert classifier.predict([1.0, 2.0]) == {"status": "model_unavailable"}


# ---------------------------------------------------------------------------
# fixture pin → SUCCESS: exact bytes flow through, real prediction
# ---------------------------------------------------------------------------

def test_pinned_bytes_activate_and_predict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = pickle.dumps(FakeModel())
    # Point the path at a NONEXISTENT file to further prove path is not the
    # load source: success must come entirely from the pinned store bytes.
    monkeypatch.setenv("CLASSIFICATION_MODEL_PATH", str(tmp_path / "nonexistent.pkl"))
    _configure_pinned_file(monkeypatch, tmp_path, data)

    classifier.load_model()
    assert isinstance(classifier._MODEL, FakeModel)
    # The exact pinned bytes flowed through (hash is over the store payload).
    assert classifier._MODEL_HASH == hashlib.sha256(data).hexdigest()[:16]

    result = classifier.predict([1.0, 2.0])
    assert result.get("status") != "model_unavailable"
    assert result["predicted_type"] == FakeModel.LABEL


# ---------------------------------------------------------------------------
# digest-tamper → degrade, never the tampered bytes
# ---------------------------------------------------------------------------

def test_digest_tamper_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = pickle.dumps(FakeModel())
    store_root = _configure_pinned_file(monkeypatch, tmp_path, original)
    # Swap the file AFTER pinning; the manifest digest still targets original.
    tampered = pickle.dumps({"not": "a model"})
    assert tampered != original
    (store_root / _RELPATH).write_bytes(tampered)

    classifier.load_model()
    assert classifier._MODEL is None  # degraded; tampered bytes never loaded
    assert classifier.predict([1.0, 2.0]) == {"status": "model_unavailable"}
