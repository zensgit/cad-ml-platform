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


def test_vectors_batch_similarity_route_is_registered_once() -> None:
    route = _matching_routes("POST", "/api/v1/vectors/similarity/batch")[0]

    assert len(_matching_routes("POST", "/api/v1/vectors/similarity/batch")) == 1
    assert getattr(route, "methods", None) == {"POST"}
    assert _matching_routes("POST", "/api/v1/vectors/similarity/batch/") == []


def test_vectors_batch_similarity_route_is_owned_by_split_router() -> None:
    assert (
        _find_route_module("POST", "/api/v1/vectors/similarity/batch")
        == "src.api.v1.vectors_similarity_router"
    )


def test_vectors_batch_similarity_route_operation_id_is_stable() -> None:
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/vectors/similarity/batch"]["post"]
    operation_ids = [
        candidate.get("operationId")
        for methods in schema["paths"].values()
        for candidate in methods.values()
        if isinstance(candidate, dict)
    ]

    assert (
        operation["operationId"]
        == "batch_similarity_api_v1_vectors_similarity_batch_post"
    )
    assert operation_ids.count(operation["operationId"]) == 1
    assert operation["summary"] == "Batch Similarity"
    assert operation["tags"] == ["向量"]
    assert set(operation["responses"]) == {"200", "422"}
    assert {
        parameter["name"]
        for parameter in operation.get("parameters", [])
        if parameter["in"] in {"query", "header"}
    } == {"X-API-Key"}
    assert operation["requestBody"]["required"] is True
    assert (
        operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/BatchSimilarityRequest"
    )
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/BatchSimilarityResponse"
    )
