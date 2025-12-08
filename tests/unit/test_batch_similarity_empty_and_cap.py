from __future__ import annotations

import os
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def _auth_headers():
    return {"X-API-Key": os.getenv("API_KEY", "test")}


def test_batch_similarity_cap_exceeded(monkeypatch):
    monkeypatch.setenv("BATCH_SIMILARITY_MAX_IDS", "2")
    payload = {"ids": ["a", "b", "c"], "top_k": 5}
    r = client.post("/api/v1/vectors/similarity/batch", json=payload, headers=_auth_headers())
    assert r.status_code == 422
    detail = r.json()["detail"] if "detail" in r.json() else r.json()
    assert detail["code"] == "INPUT_VALIDATION_FAILED"
    assert detail["stage"] == "batch_similarity"
    # batch_size and max_batch are in context sub-object
    assert detail["context"]["batch_size"] == 3
    assert detail["context"]["max_batch"] == 2


def test_batch_similarity_empty_results(monkeypatch):
    # Ensure max cap is high enough
    monkeypatch.setenv("BATCH_SIMILARITY_MAX_IDS", "10")
    # Inject a minimal in-memory vector store scenario
    from src.core import similarity as sim
    sim._VECTOR_STORE.clear()  # type: ignore
    sim._VECTOR_META.clear()  # type: ignore
    # Create two vectors sufficiently orthogonal so min_score filters everything
    sim._VECTOR_STORE["v1"] = [1.0, 0.0, 0.0]
    sim._VECTOR_META["v1"] = {"feature_version": "v1"}
    sim._VECTOR_STORE["v2"] = [0.0, 1.0, 0.0]
    sim._VECTOR_META["v2"] = {"feature_version": "v1"}
    payload = {"ids": ["v1", "v2"], "top_k": 3, "min_score": 0.99}
    r = client.post("/api/v1/vectors/similarity/batch", json=payload, headers=_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["successful"] == 2
    # Each item should have empty similar list due to strict min_score
    for it in data["items"]:
        if it["status"] == "success":
            assert it["similar"] == []
