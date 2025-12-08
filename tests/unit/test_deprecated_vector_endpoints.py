from fastapi.testclient import TestClient

from src.main import app


def test_deprecated_vector_list_endpoint():
    client = TestClient(app)
    resp = client.get("/api/v1/analyze/vectors", headers={"X-API-Key": "test"})
    # The old path /api/v1/analyze/vectors is deprecated and returns 410 Gone
    # The new path is /api/v1/vectors
    assert resp.status_code == 410
    body = resp.json()
    assert body.get("detail", {}).get("code") == "RESOURCE_GONE"


def test_deprecated_vector_delete_endpoint():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/analyze/vectors/delete",
        json={"id": "non-existent"},
        headers={"X-API-Key": "test"},
    )
    # Expect 404 structured error for non-existent vector
    if resp.status_code == 404:
        body = resp.json()
        assert body.get("detail", {}).get("code") == "DATA_NOT_FOUND"

