from src.core.similarity import FaissVectorStore


def test_faiss_backend_placeholder_behaviour():
    store = FaissVectorStore()
    if not getattr(store, "_available", False):
        # Without faiss installed, query should return empty list and get None
        assert store.query([0.1, 0.2]) == []
        assert store.get("any") is None
    else:
        # If faiss accidentally available, methods should raise NotImplementedError
        import pytest
        with pytest.raises(NotImplementedError):
            store.add("x", [0.1])
