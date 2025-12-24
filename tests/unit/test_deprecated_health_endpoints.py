from fastapi.testclient import TestClient

from src.main import app


def test_deprecated_feature_cache_endpoint_returns_410():
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/features/cache", headers={"X-API-Key": "test"})
    assert resp.status_code == 410
    data = resp.json()
    # detail is now structured error format
    detail = data["detail"]
    assert detail["code"] == "RESOURCE_GONE"
    assert "Moved" in detail["message"] or "moved" in detail["message"].lower()
    assert "new_path" in detail["context"]


def test_deprecated_faiss_health_endpoint_returns_410():
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/faiss/health", headers={"X-API-Key": "test"})
    assert resp.status_code == 410
    data = resp.json()
    # detail is now structured error format
    detail = data["detail"]
    assert detail["code"] == "RESOURCE_GONE"
    assert "Moved" in detail["message"] or "moved" in detail["message"].lower()
    assert "new_path" in detail["context"]
