import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_analyze_dxf_triggers_l2_fusion():
    # Minimal DXF structure with SECTION marker for signature validation.
    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("Bolt_M6x20.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})
    assert classification.get("part_type") == "bolt"
    assert classification.get("confidence_source") == "fusion"
    assert classification.get("rule_version") == "L2-Fusion-v1"


def test_analyze_dxf_fusion_inputs(monkeypatch):
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "false")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "false")
    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("Bolt_M6x20.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})
    fusion_decision = classification.get("fusion_decision") or {}
    fusion_inputs = classification.get("fusion_inputs") or {}
    assert fusion_decision.get("schema_version") == "v1.2"
    assert "l1" in fusion_inputs
    assert "l2" in fusion_inputs
    assert "l3" in fusion_inputs


def test_analyze_dxf_graph2d_override(monkeypatch):
    class _StubGraph2D:
        def predict_from_bytes(self, data, file_name):  # noqa: ANN001
            return {"label": "模板", "confidence": 0.92, "status": "ok"}

    monkeypatch.setenv("GRAPH2D_ENABLED", "true")
    monkeypatch.setenv("GRAPH2D_FUSION_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_ENABLED", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE", "true")
    monkeypatch.setenv("FUSION_ANALYZER_OVERRIDE_MIN_CONF", "0.5")
    monkeypatch.setenv("FUSION_GRAPH2D_OVERRIDE_LABELS", "模板")

    monkeypatch.setattr(
        "src.ml.vision_2d.get_2d_classifier",
        lambda: _StubGraph2D(),
    )
    from src.core.knowledge.fusion_analyzer import get_fusion_analyzer

    fusion = get_fusion_analyzer()
    fusion.graph2d_override_labels = {"模板"}
    fusion.graph2d_override_min_conf = 0.5
    fusion.graph2d_override_low_conf_labels = set()
    fusion.graph2d_override_low_conf_min = 0.5
    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("Template_A1.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})
    assert classification.get("part_type") == "模板"
    assert classification.get("confidence_source") == "fusion"
    assert str(classification.get("rule_version", "")).startswith("FusionAnalyzer-")
    fusion_decision = classification.get("fusion_decision") or {}
    assert fusion_decision.get("primary_label") == "模板"
    assert fusion_decision.get("source") == "hybrid"
