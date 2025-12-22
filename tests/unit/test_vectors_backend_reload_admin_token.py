from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _headers(admin_token: str | None = None):
    headers = {"X-API-Key": "test"}
    if admin_token is not None:
        headers["X-Admin-Token"] = admin_token
    return headers


def test_vectors_backend_reload_requires_admin_token():
    response = client.post("/api/v1/vectors/backend/reload", headers=_headers())
    assert response.status_code == 401


def test_vectors_backend_reload_rejects_invalid_token():
    response = client.post(
        "/api/v1/vectors/backend/reload",
        headers=_headers("invalid"),
    )
    assert response.status_code == 403


def test_vectors_backend_reload_invalid_backend():
    response = client.post(
        "/api/v1/vectors/backend/reload?backend=invalid",
        headers=_headers("test"),
    )
    assert response.status_code == 400
    detail = response.json().get("detail", {})
    assert detail.get("code") == "INPUT_VALIDATION_FAILED"


def test_vectors_backend_reload_success_memory():
    response = client.post(
        "/api/v1/vectors/backend/reload?backend=memory",
        headers=_headers("test"),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["backend"] == "memory"
