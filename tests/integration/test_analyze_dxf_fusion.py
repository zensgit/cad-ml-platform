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
