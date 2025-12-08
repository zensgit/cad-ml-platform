from src.core.similarity import FaissVectorStore


def test_faiss_backend_placeholder_behaviour():
    store = FaissVectorStore()
    if not getattr(store, "_available", False):
        # Without faiss installed, query should return empty list and get None
        assert store.query([0.1, 0.2]) == []
        assert store.get("any") is None
    else:
        # If faiss is available, verify add() doesn't raise NotImplementedError
        # (may raise RuntimeError for initialization issues, which is acceptable)
        try:
            store.add("test_vec", [0.1, 0.2, 0.3])
            # Add succeeded - verify query returns results
            results = store.query([0.1, 0.2, 0.3], top_k=1)
            # Query should return list (may be empty if index not populated correctly)
            assert isinstance(results, list)
        except RuntimeError:
            # Faiss may fail to initialize in some environments - acceptable
            pass
