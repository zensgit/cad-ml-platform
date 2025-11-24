from fastapi.testclient import TestClient

from src.main import app


def test_feature_cache_stats_endpoint(monkeypatch):
    client = TestClient(app)
    resp = client.get("/api/v1/features/cache", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "size" in data and "capacity" in data
    # hit_ratio may be None if no accesses yet
    assert "hit_ratio" in data
