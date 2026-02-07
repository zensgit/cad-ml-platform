from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_health_payload_includes_ml_config_section() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "config" in payload
    assert "ml" in payload["config"]
    assert "classification" in payload["config"]["ml"]
    assert "sampling" in payload["config"]["ml"]
    assert "hybrid_enabled" in payload["config"]["ml"]["classification"]
    assert "max_nodes" in payload["config"]["ml"]["sampling"]


def test_health_hybrid_runtime_endpoint_returns_effective_config() -> None:
    client = TestClient(app)
    response = client.get(
        "/api/v1/health/ml/hybrid-config", headers={"X-API-Key": "test"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "config" in payload
    assert "filename" in payload["config"]
    assert "graph2d" in payload["config"]
