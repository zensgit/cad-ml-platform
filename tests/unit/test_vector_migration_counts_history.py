from __future__ import annotations

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _auth_headers():
    import os
    return {"X-API-Key": os.getenv("API_KEY", "test")}


def test_vector_migrate_history_counts(monkeypatch):
    from src.core import similarity as sim
    sim._VECTOR_STORE.clear()  # type: ignore
    sim._VECTOR_META.clear()  # type: ignore
    # Prepare vectors across versions
    sim._VECTOR_STORE["a"] = [1,2,3,4,5,6,7]  # v1
    sim._VECTOR_META["a"] = {"feature_version": "v1"}
    sim._VECTOR_STORE["b"] = [1,2,3,4,5,6,7,0,0,0,0,0]  # v2 (len=12)
    sim._VECTOR_META["b"] = {"feature_version": "v2"}
    sim._VECTOR_STORE["c"] = [1,2,3,4,5,6,7]  # v1 for dry_run
    sim._VECTOR_META["c"] = {"feature_version": "v1"}
    payload = {"ids": ["a","b","missing","c"], "to_version": "v3", "dry_run": True}
    r = client.post("/api/v1/vectors/migrate", json=payload, headers=_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 4
    # Check counts in history
    r2 = client.get("/api/v1/vectors/migrate/status", headers=_auth_headers())
    assert r2.status_code == 200
    status = r2.json()
    history = status["history"]
    assert history
    last = history[-1]
    counts = last.get("counts")
    assert counts
    assert counts["dry_run"] >= 2  # a & c dry_run
    assert counts["not_found"] == 1
