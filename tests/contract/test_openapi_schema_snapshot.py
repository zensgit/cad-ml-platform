from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.main import app


def _build_openapi_contract_snapshot(schema: dict[str, Any]) -> dict[str, Any]:
    paths = schema.get("paths", {}) if isinstance(schema.get("paths"), dict) else {}
    normalized_paths: dict[str, dict[str, dict[str, Any]]] = {}
    operation_count = 0

    for path in sorted(paths):
        methods = paths[path]
        if not isinstance(methods, dict):
            continue
        method_map: dict[str, dict[str, Any]] = {}
        for method in sorted(methods):
            if method not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            operation = methods[method]
            if not isinstance(operation, dict):
                continue
            operation_count += 1
            method_map[method] = {
                "operationId": operation.get("operationId"),
                "deprecated": bool(operation.get("deprecated", False)),
                "responses": sorted(
                    [str(code) for code in (operation.get("responses") or {}).keys()]
                ),
            }
        if method_map:
            normalized_paths[path] = method_map

    components = schema.get("components", {})
    component_schemas = (
        components.get("schemas", {}) if isinstance(components, dict) else {}
    )
    raw_schema_names = (
        sorted(component_schemas.keys()) if isinstance(component_schemas, dict) else []
    )
    normalized_schema_names = []
    for name in raw_schema_names:
        if isinstance(name, str) and name.startswith("src__") and "__" in name:
            normalized_schema_names.append(name.split("__")[-1])
        else:
            normalized_schema_names.append(name)
    schema_names = sorted(set(normalized_schema_names))

    info = schema.get("info", {}) if isinstance(schema.get("info"), dict) else {}

    return {
        "openapi": schema.get("openapi"),
        "title": info.get("title"),
        "version": info.get("version"),
        "path_count": len(normalized_paths),
        "operation_count": operation_count,
        "components_schema_count": len(schema_names),
        "components_schema_names": schema_names,
        "paths": normalized_paths,
    }


def test_openapi_schema_matches_snapshot() -> None:
    baseline_path = Path("config/openapi_schema_snapshot.json")
    assert baseline_path.exists(), (
        "Missing OpenAPI snapshot baseline: "
        "config/openapi_schema_snapshot.json"
    )

    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    snapshot = _build_openapi_contract_snapshot(response.json())

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert snapshot == baseline, (
        "OpenAPI snapshot mismatch. If changes are intentional, run:\n"
        "  .venv/bin/python scripts/ci/generate_openapi_schema_snapshot.py "
        "--output config/openapi_schema_snapshot.json"
    )
