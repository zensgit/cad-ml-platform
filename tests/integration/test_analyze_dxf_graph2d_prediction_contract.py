import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def _post_minimal_dxf(file_name: str):
    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    return client.post(
        "/api/v1/analyze/",
        files={
            "file": (file_name, io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )


def test_analyze_dxf_graph2d_prediction_defaults_to_hybrid_config(monkeypatch):
    """When GRAPH2D_MIN_CONF is absent, it should default to Hybrid config (0.5)."""

    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "传动件", "confidence": 0.49, "status": "ok"}

    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ALLOW_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_EXCLUDE_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_DRAWING_TYPE_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_COARSE_LABELS", raising=False)

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("UNKNOWNv1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}

    assert graph2d.get("label") == "传动件"
    assert graph2d.get("min_confidence") == 0.5
    assert graph2d.get("passed_threshold") is False
    assert graph2d.get("excluded") is False
    assert graph2d.get("allowed") is True
    assert graph2d.get("is_drawing_type") is False
    assert graph2d.get("is_coarse_label") is True


def test_analyze_dxf_graph2d_drawing_type_flag(monkeypatch):
    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "模板", "confidence": 0.92, "status": "ok"}

    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ALLOW_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_EXCLUDE_LABELS", raising=False)

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("Template_A1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}

    assert graph2d.get("label") == "模板"
    assert graph2d.get("is_drawing_type") is True
    assert graph2d.get("is_coarse_label") is False


def test_analyze_dxf_graph2d_excluded_label_flag(monkeypatch):
    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "other", "confidence": 0.92, "status": "ok"}

    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ALLOW_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_EXCLUDE_LABELS", raising=False)

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("UNKNOWNv1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}

    assert graph2d.get("label") == "other"
    assert graph2d.get("excluded") is True


def test_analyze_dxf_graph2d_soft_override_defaults_to_graph2d_min_conf(monkeypatch):
    """When GRAPH2D_SOFT_OVERRIDE_MIN_CONF is absent, it should default to Graph2D min_conf (0.5)."""

    from src.core.analyzer import CADAnalyzer

    async def _fake_classify_part(self, doc, features):  # noqa: ANN001, ANN201
        return {
            "type": "unknown",
            "confidence": 0.2,
            "sub_type": None,
            "characteristics": [],
            "rule_version": "v1",
        }

    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "人孔", "confidence": 0.49, "status": "ok"}

    monkeypatch.setattr(CADAnalyzer, "classify_part", _fake_classify_part)
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_MIN_MARGIN", raising=False)
    monkeypatch.delenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ALLOW_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_EXCLUDE_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_DRAWING_TYPE_LABELS", raising=False)
    monkeypatch.delenv("GRAPH2D_COARSE_LABELS", raising=False)

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("UNKNOWNv1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}
    soft = cls_payload.get("soft_override_suggestion") or {}

    assert graph2d.get("min_confidence") == 0.5
    assert soft.get("threshold") == 0.5
    assert soft.get("eligible") is False
    assert soft.get("reason") == "below_threshold"


def test_analyze_dxf_graph2d_model_unavailable_still_attaches_prediction(monkeypatch):
    """Graph2D should attach `graph2d_prediction` even when model is unavailable."""

    from src.core.analyzer import CADAnalyzer

    async def _fake_classify_part(self, doc, features):  # noqa: ANN001, ANN201
        return {
            "type": "unknown",
            "confidence": 0.2,
            "sub_type": None,
            "characteristics": [],
            "rule_version": "v1",
        }

    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"status": "model_unavailable"}

    monkeypatch.setattr(CADAnalyzer, "classify_part", _fake_classify_part)
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.delenv("GRAPH2D_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_MIN_MARGIN", raising=False)
    monkeypatch.delenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", raising=False)
    monkeypatch.delenv("GRAPH2D_ENSEMBLE_ENABLED", raising=False)

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("UNKNOWNv1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}
    soft = cls_payload.get("soft_override_suggestion") or {}

    assert graph2d.get("status") == "model_unavailable"
    assert graph2d.get("min_confidence") == 0.5
    assert graph2d.get("ensemble_enabled") is False

    assert soft.get("eligible") is False
    assert soft.get("reason") == "graph2d_unavailable"
    assert soft.get("threshold") == 0.5


def test_analyze_dxf_graph2d_min_margin_blocks_soft_override(monkeypatch):
    """When GRAPH2D_MIN_MARGIN is set and margin is low, soft-override should be ineligible."""

    from src.core.analyzer import CADAnalyzer

    async def _fake_classify_part(self, doc, features):  # noqa: ANN001, ANN201
        return {
            "type": "传动件",
            "confidence": 0.2,
            "sub_type": None,
            "characteristics": [],
            "rule_version": "v1",
        }

    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {
                "label": "人孔",
                "confidence": 0.92,
                "margin": 0.01,
                "top2_confidence": 0.91,
                "status": "ok",
            }

    monkeypatch.setattr(CADAnalyzer, "classify_part", _fake_classify_part)
    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("HYBRID_CLASSIFIER_ENABLED", "false")
    monkeypatch.setenv("GRAPH2D_MIN_CONF", "0.0")
    monkeypatch.setenv("GRAPH2D_MIN_MARGIN", "0.2")
    monkeypatch.setenv("GRAPH2D_SOFT_OVERRIDE_MIN_CONF", "0.0")

    monkeypatch.setattr(
        "src.core.providers.classifier.Graph2DClassifierProviderAdapter._build_default_classifier",
        lambda self: _StubGraph2D(),
    )

    resp = _post_minimal_dxf("UNKNOWNv1.dxf")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    cls_payload = payload.get("results", {}).get("classification", {})
    graph2d = cls_payload.get("graph2d_prediction") or {}
    soft = cls_payload.get("soft_override_suggestion") or {}

    assert graph2d.get("min_margin") == 0.2
    assert graph2d.get("passed_margin") is False
    assert soft.get("eligible") is False
    assert soft.get("reason") == "below_margin"
