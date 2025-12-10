"""Vector store implementations for CAD-ML Platform.

This module provides pluggable vector storage backends with a unified interface.
Supports Qdrant (recommended) and FAISS (legacy/fallback).

Example:
    >>> from src.core.vector_stores import get_vector_store
    >>> store = get_vector_store()
    >>> await store.register_vector("doc-1", [0.1, 0.2, ...], {"material": "steel"})
    >>> results = await store.search_similar([0.1, 0.2, ...], top_k=5)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols import VectorStoreProtocol

logger = logging.getLogger(__name__)

__all__ = [
    "get_vector_store",
    "QdrantVectorStore",
    "VectorStoreError",
    "VectorStoreConnectionError",
]


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""

    pass


def get_vector_store(backend: str | None = None) -> "VectorStoreProtocol":
    """Get vector store instance with automatic fallback.

    Args:
        backend: Backend to use ("qdrant", "faiss", or None for auto-select).
                 If None, uses VECTOR_STORE_BACKEND env var, defaulting to "qdrant".

    Returns:
        VectorStoreProtocol implementation instance.

    Raises:
        VectorStoreError: If no backend is available.

    Example:
        >>> store = get_vector_store()  # Auto-select
        >>> store = get_vector_store("qdrant")  # Explicit
        >>> store = get_vector_store("faiss")  # Fallback
    """
    if backend is None:
        backend = os.getenv("VECTOR_STORE_BACKEND", "qdrant")

    if backend == "qdrant":
        try:
            from src.core.vector_stores.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore()
            # Health check
            if store.health_check():
                logger.info("Using Qdrant vector store")
                return store
            else:
                logger.warning("Qdrant health check failed, falling back to FAISS")
                backend = "faiss"
        except Exception as e:
            logger.warning(f"Qdrant unavailable ({e}), falling back to FAISS")
            backend = "faiss"

    if backend == "faiss":
        try:
            from src.core.vector_stores.faiss_store import FAISSVectorStore

            logger.info("Using FAISS vector store (fallback)")
            return FAISSVectorStore()
        except ImportError:
            # Try legacy implementation
            try:
                from src.core.similarity import (
                    register_vector,
                    search_similar,
                )

                logger.info("Using legacy FAISS implementation")
                from src.core.vector_stores.legacy_adapter import LegacyVectorStoreAdapter

                return LegacyVectorStoreAdapter()
            except ImportError:
                pass

    raise VectorStoreError(
        f"No vector store backend available. Tried: {backend}. "
        "Install qdrant-client or ensure FAISS is configured."
    )


# Lazy imports for type hints
def __getattr__(name: str):
    if name == "QdrantVectorStore":
        from src.core.vector_stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
