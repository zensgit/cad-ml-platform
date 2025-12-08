from fastapi.testclient import TestClient

from src.main import app


def test_vectors_list_endpoint():
    client = TestClient(app)
    r = client.get("/api/v1/vectors", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "total" in data and "vectors" in data


def test_vectors_update_not_found():
    client = TestClient(app)
    r = client.post(
        "/api/v1/vectors/update",
        json={"id": "nope", "replace": [1.0, 2.0]},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"


def test_vectors_migrate_dry_run():
    client = TestClient(app)
    r = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["none"], "to_version": "v2", "dry_run": True},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "items" in data

