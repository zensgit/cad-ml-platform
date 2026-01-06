from __future__ import annotations

from typing import Any, Dict, List

from src.core import similarity


def test_get_vector_from_memory() -> None:
    original_store: Dict[str, List[float]] = similarity._VECTOR_STORE
    original_backend: str = similarity._BACKEND
    original_lock = similarity._VECTOR_LOCK

    try:
        similarity._VECTOR_STORE = {"doc1": [1.0, 2.0, 3.0]}
        similarity._BACKEND = "memory"

        assert similarity.get_vector("doc1") == [1.0, 2.0, 3.0]
        assert similarity.get_vector("missing") is None
    finally:
        similarity._VECTOR_STORE = original_store
        similarity._BACKEND = original_backend
        similarity._VECTOR_LOCK = original_lock
