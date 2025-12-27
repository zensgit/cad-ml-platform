import os
import time

from fastapi.testclient import TestClient

from src.main import app


def setup_function():
    # Reset cache singleton for isolation
    from src.core.feature_cache import reset_feature_cache_for_tests

    reset_feature_cache_for_tests()
    os.environ["ADMIN_TOKEN"] = "admin_test"


client = TestClient(app)


def test_cache_apply_and_rollback_window():
    # Initial stats
    r = client.get("/api/v1/features/cache")
    assert r.status_code == 200
    cur = r.json()
    cap0 = cur["capacity"]
    ttl0 = cur["ttl_seconds"]

    # Apply new settings
    r = client.post(
        "/api/v1/features/cache/apply",
        json={"capacity": cap0 + 10, "ttl_seconds": ttl0 + 5},
        headers={"X-Admin-Token": "admin_test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "applied"
    assert data["applied"]["capacity"] == cap0 + 10
    assert data["applied"]["ttl_seconds"] == ttl0 + 5
    assert "snapshot" in data

    # Applying again within window should be rejected
    r = client.post(
        "/api/v1/features/cache/apply",
        json={"capacity": cap0 + 20},
        headers={"X-Admin-Token": "admin_test"},
    )
    assert r.status_code == 200
    data2 = r.json()
    assert data2["status"] == "window_active"
    assert data2.get("error")

    # Rollback should work
    r = client.post(
        "/api/v1/features/cache/rollback",
        headers={"X-Admin-Token": "admin_test"},
    )
    assert r.status_code == 200
    rb = r.json()
    assert rb["status"] == "rolled_back"

    # After rollback, can apply again
    r = client.post(
        "/api/v1/features/cache/apply",
        json={"capacity": cap0 + 15},
        headers={"X-Admin-Token": "admin_test"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "applied"


def test_cache_prewarm_endpoint():
    # Populate cache minimally by setting and then prewarming
    # Directly interact with cache internals to create entries
    from src.core.feature_cache import get_feature_cache

    cache = get_feature_cache()
    cache.set("k1", [0.1])
    cache.set("k2", [0.2])

    r = client.post(
        "/api/v1/features/cache/prewarm",
        params={"strategy": "auto", "limit": 1},
        headers={"X-Admin-Token": "admin_test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["touched"] == 1
    assert data["size"] >= 2
