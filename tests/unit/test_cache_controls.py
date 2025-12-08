import os
from fastapi.testclient import TestClient

from src.main import app
from src.core.feature_cache import reset_feature_cache_for_tests


def setup_function():
    reset_feature_cache_for_tests()
    # Reset ADMIN_TOKEN to default "test" for tests
    os.environ["ADMIN_TOKEN"] = "test"


def test_cache_apply_rejects_during_active_rollback_window():
    client = TestClient(app)
    # First apply to create snapshot
    r1 = client.post(
        "/api/v1/health/features/cache/apply",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={"capacity": 256, "ttl_seconds": 120},
    )
    assert r1.status_code == 200
    assert r1.json().get("status") == "applied"
    # Second apply should be rejected due to active window
    r2 = client.post(
        "/api/v1/health/features/cache/apply",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={"capacity": 512, "ttl_seconds": 300},
    )
    assert r2.status_code == 200
    assert r2.json().get("status") == "window_active"
    err = r2.json().get("error") or {}
    assert err.get("code") == "CACHE_TUNING_ROLLBACK_WINDOW_ACTIVE"


def test_cache_rollback_then_reapply_allowed():
    client = TestClient(app)
    # First apply
    r1 = client.post(
        "/api/v1/health/features/cache/apply",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={"capacity": 200, "ttl_seconds": 100},
    )
    assert r1.json().get("status") == "applied"
    # Rollback
    rb = client.post(
        "/api/v1/health/features/cache/rollback",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
    )
    assert rb.status_code == 200
    assert rb.json().get("status") in {"rolled_back", "expired", "no_snapshot"}
    # After rollback (or expired clears), re-apply should not be 'window_active'
    r2 = client.post(
        "/api/v1/health/features/cache/apply",
        headers={"X-API-Key": "test", "X-Admin-Token": "test"},
        json={"capacity": 300, "ttl_seconds": 200},
    )
    assert r2.status_code == 200
    assert r2.json().get("status") != "window_active"

