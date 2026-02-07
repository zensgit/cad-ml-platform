from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_v16_classifier_health_route_exists():
    resp = client.get("/api/v1/health/v16-classifier")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "loaded" in data


def test_v16_classifier_cache_clear_requires_admin_token():
    resp = client.post("/api/v1/health/v16-classifier/cache/clear")
    assert resp.status_code == 401


def test_v16_classifier_cache_clear_with_admin_token():
    resp = client.post(
        "/api/v1/health/v16-classifier/cache/clear",
        headers={"X-Admin-Token": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in ("ok", "unavailable", "error")


def test_v16_classifier_speed_mode_set_rejects_invalid_mode():
    resp = client.post(
        "/api/v1/health/v16-classifier/speed-mode",
        json={"speed_mode": "nope"},
        headers={"X-Admin-Token": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "error"


def test_v16_classifier_speed_mode_get_route_exists():
    resp = client.get("/api/v1/health/v16-classifier/speed-mode")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data

