from __future__ import annotations

from src.main import app


def _matching_routes(method: str, path: str) -> list[object]:
    matches: list[object] = []
    for route in app.routes:
        route_path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if route_path == path and method in methods:
            matches.append(route)
    return matches


def _find_route_module(method: str, path: str) -> str | None:
    for route in _matching_routes(method, path):
        endpoint = getattr(route, "endpoint", None)
        return getattr(endpoint, "__module__", None)
    return None


def test_vectors_list_route_is_registered_once() -> None:
    route = _matching_routes("GET", "/api/v1/vectors/")[0]

    assert len(_matching_routes("GET", "/api/v1/vectors/")) == 1
    assert getattr(route, "methods", None) == {"GET"}
    assert _matching_routes("GET", "/api/v1/vectors") == []


def test_vectors_list_route_is_owned_by_split_router() -> None:
    assert _find_route_module("GET", "/api/v1/vectors/") == "src.api.v1.vectors_list_router"


def test_vectors_list_route_operation_id_is_stable() -> None:
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/vectors/"]["get"]
    operation_ids = [
        candidate.get("operationId")
        for methods in schema["paths"].values()
        for candidate in methods.values()
        if isinstance(candidate, dict)
    ]

    assert operation["operationId"] == "list_vectors_api_v1_vectors__get"
    assert operation_ids.count(operation["operationId"]) == 1
    assert operation["summary"] == "List Vectors"
    assert operation["tags"] == ["向量"]
    assert "requestBody" not in operation
    assert set(operation["responses"]) == {"200", "422"}
    assert {
        parameter["name"]
        for parameter in operation["parameters"]
        if parameter["in"] in {"query", "header"}
    } == {
        "source",
        "offset",
        "limit",
        "material_filter",
        "complexity_filter",
        "fine_part_type_filter",
        "coarse_part_type_filter",
        "decision_source_filter",
        "is_coarse_label_filter",
        "X-API-Key",
    }
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/VectorListResponse"
    )
