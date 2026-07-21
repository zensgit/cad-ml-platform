from __future__ import annotations

import hashlib
import json
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


def _pin_single_file(
    monkeypatch,
    tmp_path: Path,
    logical_activation_id: str,
    artifact_id: str,
    data: bytes,
    relpath: str,
) -> Path:
    """Pin ``data`` as a SINGLE_FILE activation through the C4 gateway.

    Phase-A decision #3 routes every family model load through the activation
    gateway (pin + sha-256 verified). This mirrors the C2 family-test fixture:
    a server-owned store root holding the artifact plus a baseline manifest
    whose digest locks these exact bytes. Returns the store root. Callers reset
    the process-wide gateway in a finally so the config never leaks.
    """
    store_root = tmp_path / "store"
    store_root.mkdir(exist_ok=True)
    (store_root / relpath).write_bytes(data)
    manifest = tmp_path / "baseline.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "logical_activation_id": logical_activation_id,
                    "artifact_id": artifact_id,
                    "kind": "single_file",
                    "digest": hashlib.sha256(data).hexdigest(),
                    "store_relpath": relpath,
                }
            ]
        ),
        encoding="utf-8",
    )
    from src.core.model_activation import activation_gateway as gw

    monkeypatch.setenv(gw.ENV_STORE_ROOT, str(store_root))
    monkeypatch.setenv("MODEL_ACTIVATION_BASELINE_MANIFEST", str(manifest))
    gw.reset_gateway_for_tests()
    return store_root


def test_graph2d_load_error_captured_on_corrupt_checkpoint(tmp_path, monkeypatch) -> None:
    # Phase-A decision #3: raw-load-by-path removed; loading is gated by the
    # activation manifest. ``Graph2DClassifier._load_model`` no longer does
    # ``torch.load(self.model_path)`` — bytes come only from the activation
    # gateway, and a gateway degrade (data is None) is a graceful FALLBACK that
    # leaves ``_load_error`` None (readiness: "graph2d:fallback"). So a corrupt
    # file at model_path is never read and can no longer be a load error. The
    # (1a) source-capture intent is preserved against the new mechanism:
    # gateway-DELIVERED (verified) bytes that fail torch.load still populate
    # ``_load_error``.
    #   (A) corrupt file at model_path, gateway UNCONFIGURED → not read →
    #       _load_error stays None (cold), _loaded False;
    #   (B) the SAME corrupt bytes pinned → activate_file returns them →
    #       torch.load raises → _load_error IS captured.
    pytest.importorskip("torch")
    from src.core.model_activation import activation_gateway as gw
    from src.ml.vision_2d import Graph2DClassifier

    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    try:
        corrupt = _corrupt_ckpt(tmp_path, "g2d.pth")
        corrupt_bytes = corrupt.read_bytes()

        # (A) Unconfigured gateway → not read → cold, no load error.
        cold = Graph2DClassifier(model_path=str(corrupt))
        assert cold._loaded is False
        assert cold._load_error is None  # a re-added raw path load would set this

        # (B) Pin the SAME corrupt bytes → gateway returns them → load raises.
        _pin_single_file(
            monkeypatch,
            tmp_path,
            "graph2d/main",
            "main",
            corrupt_bytes,
            "graph2d_main.pth",
        )
        clf = Graph2DClassifier(model_path=str(tmp_path / "unused.pth"))
        assert clf._loaded is False
        assert clf._load_error is not None
    finally:
        gw.reset_gateway_for_tests()


def test_uvnet_load_error_captured_on_corrupt_checkpoint(tmp_path, monkeypatch) -> None:
    # Phase-A decision #3: raw-load-by-path removed; loading is gated by the
    # activation manifest. ``UVNetEncoder._load_model`` no longer does
    # ``torch.load(self.model_path)`` — bytes come only from the activation
    # gateway, and a gateway degrade (data is None) returns to the mock encoder
    # WITHOUT setting ``_load_error``. So a corrupt file at model_path is never
    # read and can no longer be a load error. The (1a) source-capture intent is
    # preserved against the new mechanism: gateway-DELIVERED (verified) bytes
    # that fail torch.load still populate ``_load_error``.
    #   (A) corrupt file at model_path, gateway UNCONFIGURED → not read →
    #       _load_error stays None (cold), _loaded False;
    #   (B) the SAME corrupt bytes pinned → activate_file returns them →
    #       torch.load raises → _load_error IS captured.
    pytest.importorskip("torch")
    from src.core.model_activation import activation_gateway as gw
    from src.ml.vision_3d import UVNetEncoder

    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    try:
        corrupt = _corrupt_ckpt(tmp_path, "uvnet.pth")
        corrupt_bytes = corrupt.read_bytes()

        # (A) Unconfigured gateway → not read → cold, no load error.
        cold = UVNetEncoder(model_path=str(corrupt))
        assert cold._loaded is False
        assert cold._load_error is None  # a re-added raw path load would set this

        # (B) Pin the SAME corrupt bytes → gateway returns them → load raises.
        _pin_single_file(
            monkeypatch,
            tmp_path,
            "vision3d-uvnet/main",
            "main",
            corrupt_bytes,
            "uvnet_main.pth",
        )
        enc = UVNetEncoder(model_path=str(tmp_path / "unused.pth"))
        assert enc._loaded is False
        assert enc._load_error is not None
    finally:
        gw.reset_gateway_for_tests()


def test_pointnet_load_error_captured_on_corrupt_checkpoint(
    tmp_path, monkeypatch
) -> None:
    # Phase-A decision #3: raw-load-by-path removed; loading is gated by the
    # activation manifest. ``PointNet3DAnalyzer._try_load_model`` no longer does
    # ``torch.load(self.model_path)`` — checkpoint bytes come only from the
    # activation gateway (pin + sha-256 verified). A corrupt file at model_path
    # is therefore NEVER read, so it can no longer be the source of a load error.
    #
    # The (1a) source-capture intent is PRESERVED against the new mechanism: a
    # load error is now produced only by gateway-DELIVERED bytes that are
    # verified-but-undecodable (an operator pinned garbage whose sha-256 the
    # manifest locks). This test proves BOTH halves, so a re-introduced raw path
    # load fails half (A):
    #   (A) corrupt file at model_path, gateway UNCONFIGURED → not read →
    #       _load_error stays None (cold), _model_loaded False;
    #   (B) the SAME corrupt bytes pinned through the gateway → activate_file
    #       returns them → torch.load raises → _load_error IS captured.
    pytest.importorskip("torch")
    from src.core.model_activation import activation_gateway as gw
    from src.ml.pointnet.inference import PointNet3DAnalyzer

    monkeypatch.delenv(gw.ENV_STORE_ROOT, raising=False)
    monkeypatch.delenv(gw.ENV_FREEZE_PARENT, raising=False)
    monkeypatch.delenv("MODEL_ACTIVATION_BASELINE_MANIFEST", raising=False)
    gw.reset_gateway_for_tests()
    try:
        corrupt = _corrupt_ckpt(tmp_path, "pointnet.pth")
        corrupt_bytes = corrupt.read_bytes()

        # (A) Unconfigured gateway → activate_file returns None → the corrupt
        #     model_path file is NEVER read → no load error, cold state.
        cold = PointNet3DAnalyzer(model_path=str(corrupt))
        assert cold._model_loaded is False
        assert cold._load_error is None  # a re-added raw path load would set this

        # (B) Pin the SAME corrupt bytes so the gateway's digest check passes and
        #     returns them; torch.load then raises and the except-capture runs.
        _pin_single_file(
            monkeypatch,
            tmp_path,
            "pointnet/main",
            "main",
            corrupt_bytes,
            "pointnet_main.pth",
        )
        analyzer = PointNet3DAnalyzer(model_path=str(tmp_path / "unused.pth"))
        assert analyzer._model_loaded is False
        assert analyzer._load_error is not None
    finally:
        gw.reset_gateway_for_tests()


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


def test_registry_surfaces_v16_load_error(monkeypatch) -> None:
    """V16 load failure is swallowed in analyzer._get_v16_classifier; the
    recorded module-level error must surface as readiness status 'error'."""
    monkeypatch.delenv("MODEL_READINESS_REQUIRED_MODELS", raising=False)
    monkeypatch.delenv("MODEL_READINESS_STRICT", raising=False)
    monkeypatch.setattr("src.core.analyzer._v16_classifier", None)
    monkeypatch.setattr(
        "src.core.analyzer._v16_classifier_load_error", "v16 boom"
    )

    item = _item(build_model_readiness_snapshot(), "v16_classifier")

    assert item["status"] == "error"
    assert item["error"] == "v16 boom"


def test_registry_surfaces_ocr_load_error(monkeypatch) -> None:
    """OCR provider bootstrap failure is recorded (and re-raised) in
    get_manager; the recorded error must surface as status 'error'."""
    monkeypatch.delenv("MODEL_READINESS_REQUIRED_MODELS", raising=False)
    monkeypatch.delenv("MODEL_READINESS_STRICT", raising=False)
    monkeypatch.setattr("src.api.v1.ocr._manager", None)
    monkeypatch.setattr("src.api.v1.ocr._manager_load_error", "ocr boom")

    item = _item(build_model_readiness_snapshot(), "ocr_provider")

    assert item["status"] == "error"
    assert item["error"] == "ocr boom"
