#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def validate_index(index_payload: Dict[str, Any], schema_payload: Dict[str, Any]) -> List[str]:
    validator = Draft202012Validator(schema_payload)
    errors: List[str] = []
    for error in sorted(validator.iter_errors(index_payload), key=lambda e: list(e.absolute_path)):
        path = ".".join([str(item) for item in list(error.absolute_path)]) or "<root>"
        errors.append(f"{path}: {error.message}")

    overview = _as_dict(index_payload.get("overview"))
    coverage = _as_dict(overview.get("artifact_coverage"))
    present = _safe_int(coverage.get("present"), -1)
    total = _safe_int(coverage.get("total"), -1)
    severity = str(overview.get("severity", "")).strip().lower()
    if present >= 0 and total >= 0 and present > total:
        errors.append("overview.artifact_coverage: present must be <= total")
    if severity == "failed" and present == total and total > 0:
        # failed is still allowed with full artifacts if upstream status failed; no strict error here.
        pass
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Graph2D context drift index json against schema."
    )
    parser.add_argument(
        "--index-json",
        required=True,
        help="Path to graph2d context drift index json.",
    )
    parser.add_argument(
        "--schema-json",
        default="config/graph2d_context_drift_index_schema.json",
        help="Path to index json schema.",
    )
    args = parser.parse_args()

    try:
        index_payload = _as_dict(_read_json(Path(str(args.index_json))))
        schema_payload = _as_dict(_read_json(Path(str(args.schema_json))))
    except Exception as exc:
        print(f"index_schema_valid=false")
        print(f"error={exc}")
        return 2

    errors = validate_index(index_payload, schema_payload)
    if errors:
        print("index_schema_valid=false")
        for item in errors:
            print(f"error={item}")
        return 3

    overview = _as_dict(index_payload.get("overview"))
    print("index_schema_valid=true")
    print(f"severity={str(overview.get('severity', ''))}")
    print(f"status={str(overview.get('status', ''))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
