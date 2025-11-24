"""Validate resilience health exposure and adaptive limiter registration."""

from fastapi.testclient import TestClient

from src.main import app


def test_health_includes_resilience_block():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "resilience" in data
    assert "status" in data["resilience"]
    assert "adaptive" in data["resilience"]


def test_adaptive_limiter_registers_on_call():
    client = TestClient(app)
    # Trigger vision analyze to register limiter via decorator
    payload = {"image_base64": "aGVsbG8=", "include_description": False}
    resp = client.post("/api/v1/vision/analyze", json=payload)
    assert resp.status_code == 200

    # Health should list adaptive_rate_limit details; guard if resilience not present
    resp2 = client.get("/health")
    data2 = resp2.json()
    res = data2.get("resilience")
    if res is not None:
        arl = res.get("adaptive_rate_limit", {})
        assert isinstance(arl, dict)
        assert len(arl.keys()) >= 1
