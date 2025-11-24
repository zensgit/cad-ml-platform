import os
from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import FaissVectorStore, register_vector

client = TestClient(app)


def test_faiss_rebuild_skipped_when_not_backend():
    os.environ["VECTOR_STORE_BACKEND"] = "memory"
    resp = client.post("/api/v1/analyze/vectors/faiss/rebuild", headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"


def test_faiss_rebuild_flow_unavailable():
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    # If faiss missing we should handle gracefully
    store = FaissVectorStore()
    register_vector("faiss_del_a", [0.1] * 7)
    if not store._available:  # type: ignore[attr-defined]
        resp = client.post("/api/v1/analyze/vectors/faiss/rebuild", headers={"x-api-key": "test"})
        assert resp.status_code == 200
        assert resp.json()["status"] in {"error", "success", "skipped"}
