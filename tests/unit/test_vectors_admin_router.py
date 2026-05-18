from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


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


def test_vectors_admin_route_is_registered_once() -> None:
    route = _matching_routes("POST", "/api/v1/vectors/backend/reload")[0]

    assert len(_matching_routes("POST", "/api/v1/vectors/backend/reload")) == 1
    assert getattr(route, "methods", None) == {"POST"}
    assert _matching_routes("POST", "/api/v1/vectors/backend/reload/") == []


def test_vectors_admin_route_is_owned_by_split_router() -> None:
    assert (
        _find_route_module("POST", "/api/v1/vectors/backend/reload")
        == "src.api.v1.vectors_admin_router"
    )


def test_vectors_admin_route_operation_id_is_stable() -> None:
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/vectors/backend/reload"]["post"]
    operation_ids = [
        candidate.get("operationId")
        for methods in schema["paths"].values()
        for candidate in methods.values()
        if isinstance(candidate, dict)
    ]

    assert (
        operation["operationId"]
        == "reload_vector_backend_api_v1_vectors_backend_reload_post"
    )
    assert operation_ids.count(operation["operationId"]) == 1
    assert operation["summary"] == "Reload Vector Backend"
    assert operation["tags"] == ["向量"]
    assert "requestBody" not in operation
    assert {
        parameter["name"]
        for parameter in operation["parameters"]
        if parameter["in"] in {"query", "header"}
    } == {"backend", "X-API-Key", "X-Admin-Token"}
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/VectorBackendReloadResponse"
    )


def test_vectors_admin_route_requires_admin_token() -> None:
    response = client.post(
        "/api/v1/vectors/backend/reload",
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 401


def test_live_vector_routes_do_not_point_back_to_facade_module() -> None:
    for route in app.routes:
        route_path = getattr(route, "path", None)
        if route_path not in {"/api/v1/vectors", "/api/v1/vectors/"} and not str(
            route_path
        ).startswith("/api/v1/vectors/"):
            continue
        endpoint = getattr(route, "endpoint", None)
        assert getattr(endpoint, "__module__", None) != "src.api.v1.vectors"
