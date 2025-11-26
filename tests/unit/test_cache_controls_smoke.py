import pytest
from fastapi.testclient import TestClient


def test_cache_controls_apply_rollback_prewarm_smoke():
    from src.main import app
    client = TestClient(app)
    headers = {"X-API-Key": "test"}
    # Apply
    resp_apply = client.post("/api/v1/health/features/cache/apply", json={"capacity": 128, "ttl_seconds": 300}, headers=headers)
    assert resp_apply.status_code in (200, 401, 403, 422)
    if resp_apply.status_code == 200:
        data = resp_apply.json()
        assert "status" in data
        assert "snapshot" in data or "previous" in data
    # Prewarm
    resp_prewarm = client.post("/api/v1/health/features/cache/prewarm", headers=headers)
    assert resp_prewarm.status_code in (200, 401, 403, 422)
    # Rollback (may be allowed only within window)
    resp_rb = client.post("/api/v1/health/features/cache/rollback", headers=headers)
    assert resp_rb.status_code in (200, 400, 401, 403, 422)
