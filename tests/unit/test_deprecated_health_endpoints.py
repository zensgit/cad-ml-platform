from fastapi.testclient import TestClient

from src.main import app


def test_deprecated_feature_cache_endpoint_returns_410():
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/features/cache", headers={"X-API-Key": "test"})
    assert resp.status_code == 410
    data = resp.json()
    assert "Moved" in data["detail"]


def test_deprecated_faiss_health_endpoint_returns_410():
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/faiss/health", headers={"X-API-Key": "test"})
    assert resp.status_code == 410
    data = resp.json()
    assert "Moved" in data["detail"]

