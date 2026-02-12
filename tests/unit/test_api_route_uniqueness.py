from __future__ import annotations

from collections import defaultdict

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
