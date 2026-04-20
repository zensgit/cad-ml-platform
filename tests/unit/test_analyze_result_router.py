from __future__ import annotations

from src.main import app


def test_analyze_result_route_owned_by_split_router() -> None:
    expected = {
        ("GET", "/api/v1/analyze/{analysis_id}"): "src.api.v1.analyze_result_router",
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
