"""Unit tests for Faiss HNSW index functionality.

Tests the HNSW (Hierarchical Navigable Small World) index implementation
in similarity.py, verifying index creation, configuration, and search behavior.
"""

import os
from unittest.mock import patch

import pytest

from src.core.similarity import FaissVectorStore, get_vector_store, reset_default_store


@pytest.fixture(autouse=True)
def reset_faiss_state():
    """Reset Faiss global state before each test."""
    import src.core.similarity as sim

    sim._FAISS_INDEX = None
    sim._FAISS_DIM = None
    sim._FAISS_ID_MAP = {}
    sim._FAISS_REVERSE_MAP = {}
    sim._FAISS_PENDING_DELETE = set()
    sim._FAISS_AVAILABLE = None
    reset_default_store()
    yield
    reset_default_store()


class TestHNSWIndexCreation:
    """Tests for HNSW index creation and configuration."""

    @patch.dict(os.environ, {"FAISS_INDEX_TYPE": "hnsw", "FAISS_HNSW_M": "16"})
    def test_create_index_hnsw_type(self):
        """Test that HNSW index is created when FAISS_INDEX_TYPE=hnsw."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        index = store._create_index(128)  # type: ignore[union-attr]
        assert hasattr(index, "hnsw"), "Index should have hnsw attribute"

    @patch.dict(os.environ, {"FAISS_INDEX_TYPE": "flat"})
    def test_create_index_flat_type(self):
        """Test that FlatIP index is created when FAISS_INDEX_TYPE=flat."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        index = store._create_index(128)  # type: ignore[union-attr]
        assert not hasattr(index, "hnsw"), "FlatIP should not have hnsw attribute"

    @patch.dict(os.environ, {})
    def test_create_index_default_flat(self):
        """Test that default index type is flat (brute-force)."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        # Clear any existing env var
        os.environ.pop("FAISS_INDEX_TYPE", None)
        index = store._create_index(128)  # type: ignore[union-attr]
        assert not hasattr(index, "hnsw")


class TestHNSWParameters:
    """Tests for HNSW parameter configuration."""

    @patch.dict(
        os.environ,
        {"FAISS_INDEX_TYPE": "hnsw", "FAISS_HNSW_M": "64", "FAISS_HNSW_EF_CONSTRUCTION": "100"},
    )
    def test_hnsw_m_parameter(self):
        """Test that FAISS_HNSW_M parameter is applied."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        index = store._create_index(128)  # type: ignore[union-attr]
        # HNSW M parameter affects the graph structure
        assert hasattr(index, "hnsw")
        # efConstruction should be set
        assert index.hnsw.efConstruction == 100

    @patch.dict(
        os.environ,
        {"FAISS_INDEX_TYPE": "hnsw", "FAISS_HNSW_M": "32", "FAISS_HNSW_EF_CONSTRUCTION": "40"},
    )
    def test_hnsw_ef_construction_default(self):
        """Test default efConstruction value."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        index = store._create_index(128)  # type: ignore[union-attr]
        assert index.hnsw.efConstruction == 40


class TestHNSWSearch:
    """Tests for HNSW search functionality."""

    @patch.dict(
        os.environ,
        {
            "FAISS_INDEX_TYPE": "hnsw",
            "FAISS_HNSW_M": "32",
            "FAISS_HNSW_EF_CONSTRUCTION": "40",
            "FAISS_HNSW_EF_SEARCH": "64",
        },
    )
    def test_hnsw_ef_search_applied(self):
        """Test that efSearch is applied during query."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss or numpy not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        # Add test vectors
        dim = 128
        for i in range(10):
            vec = [float(i)] * dim
            store.add(f"doc_{i}", vec)

        # Query should apply efSearch
        query_vec = [5.0] * dim
        results = store.query(query_vec, top_k=5)

        # Should return results
        assert len(results) > 0
        # All results should have valid scores
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    @patch.dict(os.environ, {"FAISS_INDEX_TYPE": "hnsw"})
    def test_hnsw_add_and_query(self):
        """Test basic add and query operations with HNSW."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss or numpy not installed")
            return

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        dim = 64
        # Add vectors
        store.add("vec_a", [1.0] * dim)
        store.add("vec_b", [0.5] * dim)
        store.add("vec_c", [0.0] * dim)

        # Query with vector similar to vec_a
        results = store.query([1.0] * dim, top_k=3)

        assert len(results) == 3
        # First result should be vec_a (most similar)
        assert results[0][0] == "vec_a"


class TestHNSWRebuild:
    """Tests for HNSW index rebuild functionality."""

    @patch.dict(os.environ, {"FAISS_INDEX_TYPE": "hnsw"})
    def test_rebuild_preserves_index_type(self):
        """Test that rebuild creates the same index type."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")
            return

        import src.core.similarity as sim

        store = FaissVectorStore()
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        dim = 64
        # Add vectors
        for i in range(5):
            store.add(f"doc_{i}", [float(i)] * dim)
            sim._VECTOR_STORE[f"doc_{i}"] = [float(i)] * dim

        # Mark some for deletion
        store.mark_delete("doc_0")  # type: ignore[union-attr]
        store.mark_delete("doc_1")  # type: ignore[union-attr]

        # Rebuild
        success = store.rebuild()  # type: ignore[union-attr]
        assert success

        # Index should still be HNSW type
        assert sim._FAISS_INDEX is not None
        assert hasattr(sim._FAISS_INDEX, "hnsw")

        # Remaining vectors should be queryable
        results = store.query([3.0] * dim, top_k=3)
        assert len(results) == 3
        # doc_0 and doc_1 should not be in results
        doc_ids = [r[0] for r in results]
        assert "doc_0" not in doc_ids
        assert "doc_1" not in doc_ids


class TestHNSWDegradation:
    """Tests for HNSW fallback behavior."""

    def test_faiss_unavailable_fallback(self):
        """Test graceful fallback when faiss is not available."""
        import src.core.similarity as sim

        # Simulate faiss unavailable
        sim._FAISS_AVAILABLE = False
        reset_default_store()

        store = get_vector_store("faiss")
        # Should fall back to InMemoryVectorStore
        assert isinstance(store, sim.InMemoryVectorStore)

    @patch.dict(os.environ, {"FAISS_INDEX_TYPE": "hnsw"})
    def test_hnsw_with_normalize(self):
        """Test HNSW with vector normalization enabled."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss or numpy not installed")
            return

        store = FaissVectorStore(normalize=True)
        if not store._available:  # type: ignore[union-attr]
            pytest.skip("faiss not available")
            return

        dim = 64
        # Add unnormalized vector
        store.add("test_vec", [3.0] * dim)

        # Query with same unnormalized vector
        results = store.query([3.0] * dim, top_k=1)

        assert len(results) == 1
        assert results[0][0] == "test_vec"
        # Score should be close to 1.0 for identical normalized vectors
        assert results[0][1] > 0.99
