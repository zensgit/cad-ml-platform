import os

from fastapi.testclient import TestClient

from src.core import similarity
from src.core.similarity import FaissVectorStore
from src.main import app

client = TestClient(app)


def test_faiss_backend_unavailable_graceful():
    # Reset faiss globals to avoid dimension conflicts from other tests
    similarity._FAISS_INDEX = None
    similarity._FAISS_DIM = None
    similarity._FAISS_ID_MAP = {}
    similarity._FAISS_REVERSE_MAP = {}

    # Force backend selection to faiss without library installed (assuming faiss not present in env)
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    store = FaissVectorStore()
    if not store._available:  # type: ignore[attr-defined]
        # Query should return empty list rather than raising
        assert store.query([0.1] * 7, top_k=3) == []
    else:
        # If faiss actually installed, add & query minimal path
        store.add("faiss_vec", [0.1] * 7)
        res = store.query([0.1] * 7, top_k=1)
        assert res, "Expected non-empty results when faiss available"
