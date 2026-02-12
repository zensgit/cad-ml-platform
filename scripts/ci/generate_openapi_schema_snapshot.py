#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

# Ensure repository root is importable when script runs as a file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import app


def build_openapi_contract_snapshot(schema: dict[str, Any]) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI contract snapshot baseline"
    )
    parser.add_argument(
        "--output",
        default="config/openapi_schema_snapshot.json",
        help="Output path for snapshot baseline JSON",
    )
    args = parser.parse_args()

    client = TestClient(app)
    response = client.get("/openapi.json")
    response.raise_for_status()
    snapshot = build_openapi_contract_snapshot(response.json())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OpenAPI snapshot written: {output_path.as_posix()}")
    print(f"paths={snapshot['path_count']} operations={snapshot['operation_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
