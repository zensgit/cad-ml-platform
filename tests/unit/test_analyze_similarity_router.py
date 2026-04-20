from __future__ import annotations

from src.main import app


def test_analyze_similarity_routes_owned_by_split_router() -> None:
    expected = {
        ("POST", "/api/v1/analyze/similarity"): "src.api.v1.analyze_similarity_router",
        ("POST", "/api/v1/analyze/similarity/topk"): "src.api.v1.analyze_similarity_router",
    }

    resolved: dict[tuple[str, str], str] = {}
    for route in app.routes:
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        if not methods or endpoint is None or path is None:
            continue
        for method in methods:
            key = (method, path)
            if key in expected:
                resolved[key] = endpoint.__module__

    assert resolved == expected
