from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers():
    import os

    return {"X-API-Key": os.getenv("API_KEY", "test")}


def test_vector_store_reload_endpoint(monkeypatch):
    # Simulate backend env change
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")
    r = client.post("/api/v1/maintenance/vectors/backend/reload", headers=_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["backend"] == "memory"
