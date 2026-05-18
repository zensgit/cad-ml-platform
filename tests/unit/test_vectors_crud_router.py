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


def test_vectors_crud_routes_are_owned_by_split_router() -> None:
    expected = {
        ("POST", "/api/v1/vectors/delete"): "src.api.v1.vectors_crud_router",
        ("POST", "/api/v1/vectors/register"): "src.api.v1.vectors_crud_router",
        ("POST", "/api/v1/vectors/search"): "src.api.v1.vectors_crud_router",
    }

    for (method, path), module in expected.items():
        assert _find_route_module(method, path) == module
