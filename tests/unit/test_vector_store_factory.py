"""Tests for VectorStore factory and dependency injection."""

import os
from unittest.mock import patch

import pytest

from src.core.similarity import (
    FaissVectorStore,
    InMemoryVectorStore,
    VectorStoreProtocol,
    get_vector_store,
    reset_default_store,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset default store before each test."""
    reset_default_store()
    yield
    reset_default_store()


def test_get_vector_store_memory_backend():
    """Test factory returns InMemoryVectorStore for memory backend."""
    store = get_vector_store(backend="memory")
    assert isinstance(store, InMemoryVectorStore)
    assert isinstance(store, VectorStoreProtocol)


def test_get_vector_store_faiss_backend_fallback():
    """Test factory falls back to InMemoryVectorStore when Faiss unavailable."""
    # Faiss may or may not be available, but factory should never raise
    store = get_vector_store(backend="faiss")
    assert isinstance(store, (InMemoryVectorStore, FaissVectorStore))
    assert isinstance(store, VectorStoreProtocol)


def test_get_vector_store_uses_env_var():
    """Test factory reads VECTOR_STORE_BACKEND environment variable."""
    with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "memory"}):
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorStore)


def test_get_vector_store_caches_default():
    """Test factory caches the default store instance."""
    store1 = get_vector_store(backend="memory")
    store2 = get_vector_store(backend="memory")
    # Should return same instance
    assert store1 is store2


def test_get_vector_store_different_backends():
    """Test factory creates different instances for different backends."""
    store1 = get_vector_store(backend="memory")
    reset_default_store()
    store2 = get_vector_store(backend="faiss")
    # Should be different instances
    assert store1 is not store2


def test_reset_default_store():
    """Test reset_default_store clears cached instance."""
    store1 = get_vector_store(backend="memory")
    reset_default_store()
    store2 = get_vector_store(backend="memory")
    # Should be different instances after reset
    assert store1 is not store2


def test_vector_store_protocol_methods():
    """Test that returned store implements VectorStoreProtocol."""
    store = get_vector_store(backend="memory")

    # Check all protocol methods exist
    assert hasattr(store, "add")
    assert hasattr(store, "get")
    assert hasattr(store, "exists")
    assert hasattr(store, "query")
    assert hasattr(store, "meta")

    # Check methods are callable
    assert callable(store.add)
    assert callable(store.get)
    assert callable(store.exists)
    assert callable(store.query)
    assert callable(store.meta)


def test_vector_store_basic_operations():
    """Test basic vector store operations through factory."""
    store = get_vector_store(backend="memory")

    # Add a vector
    test_vector = [1.0, 2.0, 3.0]
    store.add("test_key", test_vector)

    # Check existence
    assert store.exists("test_key")
    assert not store.exists("nonexistent_key")

    # Retrieve vector
    retrieved = store.get("test_key")
    assert retrieved == test_vector

    # Query (should return self as most similar)
    results = store.query(test_vector, top_k=1)
    assert len(results) > 0


def test_get_vector_store_explicit_backend_overrides_env():
    """Test that explicit backend parameter overrides environment variable."""
    with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "faiss"}):
        store = get_vector_store(backend="memory")
        # Should use explicit backend, not env var
        assert isinstance(store, InMemoryVectorStore)


def test_get_vector_store_default_is_memory():
    """Test that default backend is memory when no env var set."""
    with patch.dict(os.environ, {}, clear=True):
        if "VECTOR_STORE_BACKEND" in os.environ:
            del os.environ["VECTOR_STORE_BACKEND"]
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorStore)


def test_factory_graceful_degradation():
    """Test factory degrades gracefully from faiss to memory on error."""
    # This test verifies the fallback mechanism
    # If Faiss is unavailable or fails to initialize, should get InMemoryVectorStore
    store = get_vector_store(backend="faiss")
    # Should always succeed, either Faiss or InMemory
    assert store is not None
    assert isinstance(store, VectorStoreProtocol)


def test_multiple_backend_switches():
    """Test switching between backends multiple times."""
    # First memory
    store1 = get_vector_store(backend="memory")
    assert isinstance(store1, InMemoryVectorStore)

    # Switch to faiss
    reset_default_store()
    store2 = get_vector_store(backend="faiss")
    assert store2 is not store1

    # Back to memory
    reset_default_store()
    store3 = get_vector_store(backend="memory")
    assert isinstance(store3, InMemoryVectorStore)
    assert store3 is not store1  # New instance


def test_protocol_type_checking():
    """Test that store satisfies VectorStoreProtocol type."""
    store = get_vector_store()

    # This would fail type checking if protocol not satisfied
    def accepts_protocol(s: VectorStoreProtocol) -> None:
        pass

    # Should not raise
    accepts_protocol(store)
