from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.main import app


def test_api_routes_have_no_duplicate_method_path_pairs() -> None:
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)

    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        name = getattr(route, "name", "<unknown>")
        if not path:
            continue
        for method in sorted(methods):
            if method in {"HEAD", "OPTIONS"}:
                continue
            key = (method, path)
            seen[key].append(name)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    assert not duplicates, f"Duplicate method/path routes found: {duplicates}"


def _find_route_module(method: str, path: str) -> Optional[str]:
    for route in app.routes:
        route_path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if route_path != path or method not in methods:
            continue
        endpoint = getattr(route, "endpoint", None)
        return getattr(endpoint, "__module__", None)
    return None


def test_critical_analyze_paths_are_owned_by_split_routers() -> None:
    expected = {
        ("GET", "/api/v1/analyze/drift"): "src.api.v1.drift",
        ("POST", "/api/v1/analyze/drift/reset"): "src.api.v1.drift",
        ("GET", "/api/v1/analyze/drift/baseline/status"): "src.api.v1.drift",
        ("GET", "/api/v1/analyze/process/rules/audit"): "src.api.v1.process",
    }

    for (method, path), module in expected.items():
        actual = _find_route_module(method, path)
        assert actual == module, (
            f"Unexpected route owner for {method} {path}: expected {module}, got {actual}"
        )
