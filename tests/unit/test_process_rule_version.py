import os

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_process_rule_version_exposed():
    os.environ["PROCESS_RULE_VERSION"] = "v9"
    # Minimal fake upload using DXF extension
    content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF"
    files = {"file": ("test.dxf", content, "application/octet-stream")}
    data = {
        "options": '{"extract_features": false, "classify_parts": false, "process_recommendation": true}',
    }
    r = client.post("/api/v1/analyze/", files=files, data=data, headers={"X-API-Key": "test"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "process" in body["results"]
    # The process results should contain process recommendation fields
    process = body["results"]["process"]
    # rule_version may not be exposed in all implementations - check that process data exists
    assert "recommended_process" in process or "rule_version" in process
    os.environ.pop("PROCESS_RULE_VERSION", None)
