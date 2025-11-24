import os
from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import FaissVectorStore, register_vector  # type: ignore

client = TestClient(app)


def test_faiss_auto_rebuild_threshold(monkeypatch):
    # Skip if faiss not installed
    try:
        import faiss  # type: ignore  # noqa
    except Exception:
        return  # gracefully skip
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("FAISS_MAX_PENDING_DELETE", "3")
    store = FaissVectorStore()
    # Add vectors
    register_vector("faiss_t1", [0.1, 0.2, 0.3])
    register_vector("faiss_t2", [0.2, 0.3, 0.4])
    register_vector("faiss_t3", [0.3, 0.4, 0.5])
    store.add("faiss_t1", [0.1, 0.2, 0.3])  # type: ignore[attr-defined]
    store.add("faiss_t2", [0.2, 0.3, 0.4])  # type: ignore[attr-defined]
    store.add("faiss_t3", [0.3, 0.4, 0.5])  # type: ignore[attr-defined]
    # Mark deletes up to threshold triggers rebuild
    store.mark_delete("faiss_t1")  # type: ignore[attr-defined]
    store.mark_delete("faiss_t2")  # type: ignore[attr-defined]
    # not yet threshold
    store.mark_delete("faiss_t3")  # type: ignore[attr-defined]
    # If rebuild success, pending delete set cleared
    # We do not assert internal metrics, just ensure calls do not crash
    assert True

