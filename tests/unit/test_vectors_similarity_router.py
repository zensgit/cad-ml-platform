from __future__ import annotations

from src.main import app


def _find_route_module(method: str, path: str) -> str | None:
    for route in app.routes:
        route_path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if route_path != path or method not in methods:
            continue
        endpoint = getattr(route, "endpoint", None)
        return getattr(endpoint, "__module__", None)
    return None


def test_vectors_batch_similarity_route_is_owned_by_split_router() -> None:
    assert (
        _find_route_module("POST", "/api/v1/vectors/similarity/batch")
        == "src.api.v1.vectors_similarity_router"
    )

