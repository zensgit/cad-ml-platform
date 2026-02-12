import io
import json
import os

from fastapi.testclient import TestClient

from src.main import app
from src.core.analyzer import CADAnalyzer
from src.core.knowledge.fusion import FusionClassifier


client = TestClient(app)


def test_analyze_dxf_hybrid_auto_override_applies_for_placeholder_rules():
    # Minimal DXF structure with SECTION marker for signature validation.
    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("J2925001-01人孔v2.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})

    assert classification.get("part_type") == "人孔"
    assert classification.get("confidence_source") == "hybrid"
    assert classification.get("rule_version") == "HybridClassifier-v1"

    applied = classification.get("hybrid_override_applied") or {}
    assert applied.get("mode") == "auto"


def test_analyze_dxf_hybrid_auto_override_applies_for_low_confidence_base(monkeypatch):
    async def _fake_classify_part(self, doc, features):
        return {
            "type": "传动件",
            "confidence": 0.1,
            "sub_type": None,
            "characteristics": [],
            "rule_version": "v2",
        }

    monkeypatch.setattr(CADAnalyzer, "classify_part", _fake_classify_part)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("J2925001-01人孔v2.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})

    assert classification.get("part_type") == "人孔"
    assert classification.get("confidence_source") == "hybrid"
    assert classification.get("rule_version") == "HybridClassifier-v1"

    applied = classification.get("hybrid_override_applied") or {}
    assert applied.get("mode") == "auto_low_conf"


def test_analyze_dxf_hybrid_auto_override_applies_for_drawing_type_base(monkeypatch):
    def _fake_fusion_classify(self, text_signals, features_2d, features_3d):
        return {
            "type": "机械制图",
            "confidence": 0.9,
            "alternatives": [],
            "fusion_breakdown": {"source": "test"},
        }

    monkeypatch.setattr(FusionClassifier, "classify", _fake_fusion_classify)

    dxf_payload = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    options = {"extract_features": True, "classify_parts": True}
    resp = client.post(
        "/api/v1/analyze/",
        files={
            "file": ("J2925001-01人孔v2.dxf", io.BytesIO(dxf_payload), "application/dxf"),
        },
        data={"options": json.dumps(options)},
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    classification = data.get("results", {}).get("classification", {})

    assert classification.get("part_type") == "人孔"
    assert classification.get("confidence_source") == "hybrid"
    assert classification.get("rule_version") == "HybridClassifier-v1"

    applied = classification.get("hybrid_override_applied") or {}
    assert applied.get("mode") == "auto_drawing_type"
