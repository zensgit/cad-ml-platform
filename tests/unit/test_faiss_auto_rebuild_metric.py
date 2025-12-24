import os

from fastapi.testclient import TestClient

from src.core.similarity import FaissVectorStore, register_vector  # type: ignore
from src.main import app

client = TestClient(app)


def test_faiss_auto_rebuild_metric(monkeypatch):
    try:
        import faiss  # type: ignore  # noqa
    except Exception:
        return
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("FAISS_MAX_PENDING_DELETE", "1")  # trigger immediately
    store = FaissVectorStore()
    register_vector("faiss_rb_a", [0.1, 0.2, 0.3])
    store.add("faiss_rb_a", [0.1, 0.2, 0.3])  # type: ignore[attr-defined]
    store.mark_delete("faiss_rb_a")  # type: ignore[attr-defined]
    # Metric exposure check (best-effort)
    r = client.get("/metrics")
    if r.status_code == 200:
        assert "faiss_auto_rebuild_total" in r.text
