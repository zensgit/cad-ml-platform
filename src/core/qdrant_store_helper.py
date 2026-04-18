from __future__ import annotations

import os


def get_qdrant_store_or_none():
    if os.getenv("VECTOR_STORE_BACKEND", "memory") != "qdrant":
        return None
    try:
        from src.core.vector_stores import get_vector_store as get_managed_vector_store

        return get_managed_vector_store("qdrant")
    except Exception:
        return None
