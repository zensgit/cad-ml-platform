"""C2 wiring discriminator: pointnet/main via the Phase-A activation gateway.

Runnable in isolation (the repo-global conftest fails to collect on this box)::

    cd /private/tmp/cadml-c2c6
    PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_c2_pointnet.py -q

Proves the C2 wiring contract for ``src/ml/pointnet/inference.py``:
  * pin-absent (no store configured) -> the family degrades, no raw load ever
    happens (torch.load is never called against a filesystem path);
  * a fixture pin (tmp file + sha256 + baseline manifest) -> the analyzer
    loads a real model and returns "ok" classification/feature status;
  * digest-tamper (bytes swapped after pinning) -> degrades, never uses the
    tampered checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.core.model_activation import activation_gateway as gw  # noqa: E402
from src.ml.pointnet.inference import PointNet3DAnalyzer  # noqa: E402
from src.ml.pointnet.model import PointNetClassifier  # noqa: E402

_LID = "pointnet/main"
_AID = "main"
_RELPATH = "pointnet_main.pt"


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


def _fake_checkpoint_bytes(num_classes: int = 8) -> bytes:
    """A tiny real PointNetClassifier state dict, serialized like a checkpoint."""
    classifier = PointNetClassifier(num_classes=num_classes)
    checkpoint = {"classifier_state_dict": classifier.state_dict()}
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    return buf.getvalue()


def _configure_pinned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, data: bytes
) -> Path:
    """Create store_root with a pinned pointnet/main file + manifest; wire env."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / _RELPATH).write_bytes(data)

    manifest_entries = [
        {
            "logical_activation_id": _LID,
            "artifact_id": _AID,
            "kind": "single_file",
            "digest": _sha256_hex(data),
            "store_relpath": _RELPATH,
        }
    ]
    manifest_path = tmp_path / "baseline.json"
    manifest_path.write_text(json.dumps(manifest_entries), encoding="utf-8")

    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest_path))
    gw.reset_gateway_for_tests()
    return store_root


def _tiny_points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((32, 3)).astype(np.float32)


def _write_xyz(path: Path, points: np.ndarray) -> None:
    lines = [f"{p[0]} {p[1]} {p[2]}" for p in points]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Pin-absent -> family degrades, no raw load
# ---------------------------------------------------------------------------


def test_pin_absent_degrades_no_raw_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """No store configured (default NO-PIN posture) -> analyzer stays degraded.

    Also asserts torch.load is never invoked -- there is no raw-load-by-path
    fallback left in the wired code path.
    """
    calls = []
    monkeypatch.setattr(
        torch,
        "load",
        lambda *a, **k: calls.append((a, k)) or pytest.fail("raw torch.load must not run"),
    )

    analyzer = PointNet3DAnalyzer(model_path="/nonexistent/whatever.pt")

    assert analyzer._model_loaded is False
    assert analyzer._classifier is None
    assert calls == []

    # Exercise the public degrade path too.
    fallback = analyzer._fallback_classification()
    assert fallback["status"] == "model_unavailable"


# ---------------------------------------------------------------------------
# 2. Fixture pin -> success path uses the exact activated bytes
# ---------------------------------------------------------------------------


def test_fixture_pin_activates_and_loads_real_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = _fake_checkpoint_bytes(num_classes=8)
    _configure_pinned_file(monkeypatch, tmp_path, data)

    analyzer = PointNet3DAnalyzer(model_path="/ignored/because/gateway/decides")

    assert analyzer._model_loaded is True
    assert analyzer._classifier is not None
    assert analyzer._load_error is None

    xyz_path = tmp_path / "part.xyz"
    _write_xyz(xyz_path, _tiny_points())

    result = analyzer.classify(str(xyz_path))
    assert result["status"] == "ok"
    assert result["label"] in analyzer.labels

    feat_result = analyzer.extract_features(str(xyz_path))
    assert feat_result["status"] == "ok"
    assert feat_result["dimension"] == analyzer.feature_dim


# ---------------------------------------------------------------------------
# 3. Digest-tamper -> degrades, never uses the tampered checkpoint
# ---------------------------------------------------------------------------


def test_digest_tamper_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = _fake_checkpoint_bytes(num_classes=8)
    store_root = _configure_pinned_file(monkeypatch, tmp_path, original)

    # Tamper the on-disk artifact AFTER pinning (digest still targets original).
    tampered = _fake_checkpoint_bytes(num_classes=8) + b"\x00tamper"
    assert tampered != original
    (store_root / _RELPATH).write_bytes(tampered)

    analyzer = PointNet3DAnalyzer(model_path="/ignored")

    assert analyzer._model_loaded is False
    assert analyzer._classifier is None

    xyz_path = tmp_path / "part2.xyz"
    _write_xyz(xyz_path, _tiny_points())
    result = analyzer.classify(str(xyz_path))
    assert result["status"] == "model_unavailable"
