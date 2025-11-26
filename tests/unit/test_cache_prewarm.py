from fastapi.testclient import TestClient

from src.main import app


def test_cache_prewarm_endpoint_basic():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/health/features/cache/prewarm",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("touched", 0), int)
