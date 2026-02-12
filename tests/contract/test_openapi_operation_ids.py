from __future__ import annotations

import warnings

from fastapi.testclient import TestClient

from src.main import app


def test_openapi_operation_ids_are_unique() -> None:
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    operation_ids: list[str] = []
    for methods in schema.get("paths", {}).values():
        if not isinstance(methods, dict):
            continue
        for operation in methods.values():
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if isinstance(operation_id, str) and operation_id:
                operation_ids.append(operation_id)

    assert operation_ids, "OpenAPI schema should expose operationIds"
    unique_ids = set(operation_ids)
    assert len(unique_ids) == len(operation_ids), (
        f"Duplicate operationIds found: total={len(operation_ids)}, unique={len(unique_ids)}"
    )


def test_openapi_generation_has_no_duplicate_operation_id_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200

    duplicate_warnings = [
        str(item.message)
        for item in caught
        if "Duplicate Operation ID" in str(item.message)
    ]
    assert not duplicate_warnings, (
        "Unexpected duplicate operation-id warnings: "
        + "; ".join(duplicate_warnings)
    )
