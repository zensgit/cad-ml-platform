from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.readiness_registry import build_model_readiness_snapshot


def _item(snapshot, name: str) -> dict:
    return snapshot.to_dict()["items"][name]


def test_missing_local_checkpoints_are_degraded_fallbacks(monkeypatch, tmp_path) -> None:
    missing_graph2d = tmp_path / "missing_graph2d.pth"
    missing_uvnet = tmp_path / "missing_uvnet.pth"
    missing_embedding = tmp_path / "missing_embeddings"
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(missing_graph2d))
    monkeypatch.setenv("UVNET_MODEL_PATH", str(missing_uvnet))
    monkeypatch.setenv("DOMAIN_EMBEDDING_MODEL_PATH", str(missing_embedding))
    monkeypatch.delenv("MODEL_READINESS_REQUIRED_MODELS", raising=False)
    monkeypatch.delenv("MODEL_READINESS_STRICT", raising=False)

    snapshot = build_model_readiness_snapshot()

    assert snapshot.ok is True
    assert snapshot.degraded is True
    assert _item(snapshot, "graph2d")["loaded"] is False
    assert _item(snapshot, "graph2d")["checkpoint_exists"] is False
    assert _item(snapshot, "graph2d")["status"] == "fallback"
    assert _item(snapshot, "embedding_model")["fallback_mode"] == "tfidf_fallback"


def test_checkpoint_presence_reports_available_and_checksum(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "graph2d.pth"
    checkpoint.write_bytes(b"graph2d-test-checkpoint")
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(checkpoint))

    snapshot = build_model_readiness_snapshot()
    graph2d = _item(snapshot, "graph2d")

    assert graph2d["checkpoint_exists"] is True
    assert graph2d["loaded"] is False
    assert graph2d["status"] == "available"
    assert isinstance(graph2d["checksum"], str)
    assert len(graph2d["checksum"]) == 16


def test_required_missing_model_blocks_readiness(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(tmp_path / "missing.pth"))
    monkeypatch.setenv("MODEL_READINESS_REQUIRED_MODELS", "graph2d")

    snapshot = build_model_readiness_snapshot()

    assert snapshot.ok is False
    assert snapshot.status == "not_ready"
    assert "graph2d:fallback" in snapshot.blocking_reasons


def test_loader_readiness_check_honors_legacy_models_loaded_patch() -> None:
    from src.models.loader import models_readiness_check

    with patch("src.models.loader.models_loaded", return_value=False):
        result = models_readiness_check()

    assert result["ok"] is False
    assert "model readiness failed" in result["detail"]


def test_health_payload_exposes_model_registry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(tmp_path / "missing.pth"))

    from src.api.health_utils import build_health_payload

    payload = build_health_payload()
    readiness = payload["config"]["ml"]["readiness"]

    assert readiness["model_registry_status"] in {"degraded", "not_ready", "ready"}
    assert "model_registry" in readiness
    assert "graph2d" in readiness["model_registry"]["items"]


def test_model_readiness_health_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(tmp_path / "missing.pth"))
    client = TestClient(app)

    response = client.get(
        "/api/v1/health/model-readiness",
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is True
    assert data["items"]["graph2d"]["status"] == "fallback"


# --- Phase 2 residual: load-error contract (graph2d / uvnet / pointnet) -----
#
# Two patterns per model, deliberately:
#  (1) source-capture: a fresh instance built against a corrupt checkpoint
#      must populate `_load_error` (validates the `except` capture in 1a).
#  (2) registry-surface: a stub singleton carrying `_load_error` must make
#      the readiness item report status="error" (validates the registry
#      threading in 1b). The import-time module singletons are NOT reloaded
#      per test, so a corrupt env path alone cannot exercise (2).


def _corrupt_ckpt(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"not-a-torch-checkpoint")
    return p


def test_graph2d_load_error_captured_on_corrupt_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.ml.vision_2d import Graph2DClassifier

    clf = Graph2DClassifier(model_path=str(_corrupt_ckpt(tmp_path, "g2d.pth")))

    assert clf._loaded is False
    assert clf._load_error is not None  # cold state would be None


def test_uvnet_load_error_captured_on_corrupt_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.ml.vision_3d import UVNetEncoder

    enc = UVNetEncoder(model_path=str(_corrupt_ckpt(tmp_path, "uvnet.pth")))

    assert enc._loaded is False
    assert enc._load_error is not None


def test_pointnet_load_error_captured_on_corrupt_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.ml.pointnet.inference import PointNet3DAnalyzer

    analyzer = PointNet3DAnalyzer(
        model_path=str(_corrupt_ckpt(tmp_path, "pointnet.pth"))
    )

    assert analyzer._model_loaded is False
    assert analyzer._load_error is not None


def test_registry_surfaces_load_error_status(monkeypatch) -> None:
    """A stub singleton carrying _load_error -> readiness status == 'error'."""
    monkeypatch.delenv("MODEL_READINESS_REQUIRED_MODELS", raising=False)
    monkeypatch.delenv("MODEL_READINESS_STRICT", raising=False)
    monkeypatch.setattr(
        "src.ml.vision_2d._graph2d",
        types.SimpleNamespace(_loaded=False, _load_error="g2d boom"),
    )
    monkeypatch.setattr(
        "src.ml.vision_3d._encoder",
        types.SimpleNamespace(_loaded=False, _load_error="uvnet boom"),
    )
    monkeypatch.setattr(
        "src.api.v1.pointcloud._analyzer",
        types.SimpleNamespace(_model_loaded=False, _load_error="pointnet boom"),
    )

    snapshot = build_model_readiness_snapshot()

    for name, msg in (
        ("graph2d", "g2d boom"),
        ("uvnet", "uvnet boom"),
        ("pointnet", "pointnet boom"),
    ):
        item = _item(snapshot, name)
        assert item["status"] == "error", name
        assert item["error"] == msg, name
        assert item["loaded"] is False, name


def test_registry_distinguishes_error_from_cold(monkeypatch, tmp_path) -> None:
    """Cold (no _load_error) must NOT be reported as 'error'."""
    monkeypatch.delenv("MODEL_READINESS_REQUIRED_MODELS", raising=False)
    monkeypatch.delenv("MODEL_READINESS_STRICT", raising=False)
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_MODEL_PATH", str(tmp_path / "missing.pth"))
    monkeypatch.setattr(
        "src.ml.vision_2d._graph2d",
        types.SimpleNamespace(_loaded=False, _load_error=None),
    )

    graph2d = _item(build_model_readiness_snapshot(), "graph2d")

    assert graph2d["error"] is None
    assert graph2d["status"] != "error"
    assert graph2d["status"] == "fallback"
