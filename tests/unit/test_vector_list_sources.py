from __future__ import annotations

from src.api.v1 import vectors as vectors_module
from src.core.vector_list_sources import resolve_vector_list_source


def test_resolve_vector_list_source_auto_uses_available_backend() -> None:
    assert resolve_vector_list_source("auto", "redis") == "redis"
    assert resolve_vector_list_source("auto", "qdrant") == "qdrant"
    assert resolve_vector_list_source("auto", "memory") == "memory"
    assert resolve_vector_list_source("auto", "unexpected") == "memory"


def test_resolve_vector_list_source_preserves_explicit_source() -> None:
    assert resolve_vector_list_source("redis", "memory") == "redis"
    assert resolve_vector_list_source("qdrant", "memory") == "qdrant"
    assert resolve_vector_list_source("memory", "redis") == "memory"


def test_vectors_facade_preserves_list_source_export() -> None:
    assert vectors_module._resolve_list_source is resolve_vector_list_source
