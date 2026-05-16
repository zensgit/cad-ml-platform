from __future__ import annotations


def resolve_vector_list_source(source: str, backend: str) -> str:
    if source == "auto":
        if backend == "redis":
            return "redis"
        if backend == "qdrant":
            return "qdrant"
        return "memory"
    return source


__all__ = ["resolve_vector_list_source"]
