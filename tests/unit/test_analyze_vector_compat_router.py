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


def test_analyze_vector_compat_routes_are_owned_by_split_router() -> None:
    expected = {
        ("POST", "/api/v1/analyze/vectors/faiss/rebuild"): "src.api.v1.analyze_vector_compat",
        ("POST", "/api/v1/analyze/vectors/update"): "src.api.v1.analyze_vector_compat",
        ("POST", "/api/v1/analyze/vectors/migrate"): "src.api.v1.analyze_vector_compat",
        ("GET", "/api/v1/analyze/vectors/migrate/status"): "src.api.v1.analyze_vector_compat",
    }

    for (method, path), module in expected.items():
        assert _find_route_module(method, path) == module
