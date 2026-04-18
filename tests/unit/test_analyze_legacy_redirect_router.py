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


def test_analyze_legacy_redirect_routes_are_owned_by_split_router() -> None:
    expected = {
        ("GET", "/api/v1/analyze/vectors/distribution"): "src.api.v1.analyze_legacy_redirects",
        ("POST", "/api/v1/analyze/vectors/delete"): "src.api.v1.analyze_legacy_redirects",
        ("GET", "/api/v1/analyze/vectors/stats"): "src.api.v1.analyze_legacy_redirects",
        ("GET", "/api/v1/analyze/features/cache"): "src.api.v1.analyze_legacy_redirects",
        ("GET", "/api/v1/analyze/faiss/health"): "src.api.v1.analyze_legacy_redirects",
    }

    for (method, path), module in expected.items():
        assert _find_route_module(method, path) == module
