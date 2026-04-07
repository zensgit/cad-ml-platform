#!/usr/bin/env python3
"""Validate Hybrid superpass report structure and related input payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REQUIRED_SUPERPASS_FIELDS: Dict[str, type] = {
    "status": str,
    "headline": str,
    "thresholds": dict,
    "checks": list,
    "failures": list,
    "warnings": list,
}

SUPERPASS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["status", "headline", "thresholds", "checks", "failures", "warnings"],
    "properties": {
        "status": {"type": "string"},
        "headline": {"type": "string"},
        "thresholds": {"type": "object"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "passed"],
                "properties": {
                    "name": {"type": "string"},
                    "passed": {"type": "boolean"},
                },
            },
        },
        "failures": {"type": "array"},
        "warnings": {"type": "array"},
    },
}

HYBRID_BLIND_GATE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "object",
            "properties": {
                "hybrid_accuracy": {"type": "number"},
                "hybrid_gain_vs_graph2d": {"type": "number"},
            },
        }
    },
}

HYBRID_CALIBRATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metrics_after": {
            "type": "object",
            "properties": {"ece": {"type": "number"}},
        }
    },
}


def _schema_path(error: Any) -> str:
    path_items = [str(item) for item in list(getattr(error, "path", []))]
    return ".".join(path_items) if path_items else "<root>"


def _validate_with_json_schema(
    *,
    payload: Optional[Dict[str, Any]],
    label: str,
    schema: Dict[str, Any],
    schema_mode: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if payload is None:
        return
    if schema_mode != "builtin":
        return
    try:
        from jsonschema import Draft7Validator  # type: ignore
    except Exception:
        warnings.append("jsonschema package unavailable, skip schema checks")
        return
    validator = Draft7Validator(schema)
    for item in validator.iter_errors(payload):
        message = str(getattr(item, "message", "schema validation error")).strip()
        errors.append(f"{label} schema violation at {_schema_path(item)}: {message}")


def _read_json_object(
    path: Path, *, label: str, errors: list[str]
) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        errors.append(f"{label} file not found: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"failed to parse {label} json {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object: {path}")
        return None
    return payload


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _validate_required_fields(
    payload: Dict[str, Any],
    *,
    field_types: Dict[str, type],
    label: str,
    errors: list[str],
) -> None:
    for key, expected_type in field_types.items():
        value = payload.get(key)
        if key not in payload:
            errors.append(f"{label}.{key} is required")
            continue
        if not isinstance(value, expected_type):
            errors.append(f"{label}.{key} must be {expected_type.__name__}")


def _validate_superpass_payload(
    payload: Dict[str, Any],
    *,
    errors: list[str],
    warnings: list[str],
    summary: Dict[str, Any],
) -> None:
    _validate_required_fields(
        payload, field_types=REQUIRED_SUPERPASS_FIELDS, label="superpass", errors=errors
    )

    checks = payload.get("checks")
    if isinstance(checks, list):
        summary["superpass_check_count"] = len(checks)
        if not checks:
            warnings.append("superpass.checks is empty")
        for idx, item in enumerate(checks):
            if not isinstance(item, dict):
                errors.append(f"superpass.checks[{idx}] must be object")
                continue
            if "name" not in item:
                errors.append(f"superpass.checks[{idx}].name is required")
            if "passed" not in item:
                errors.append(f"superpass.checks[{idx}].passed is required")

    status = str(payload.get("status", "")).strip().lower()
    if status and status not in {"passed", "failed"}:
        warnings.append(
            f"superpass.status has unexpected value: {payload.get('status')!r}"
        )

    summary["superpass_status"] = payload.get("status")
    summary["superpass_failure_count"] = (
        len(payload.get("failures"))
        if isinstance(payload.get("failures"), list)
        else None
    )
    summary["superpass_warning_count"] = (
        len(payload.get("warnings"))
        if isinstance(payload.get("warnings"), list)
        else None
    )


def _validate_hybrid_blind_gate_payload(
    payload: Optional[Dict[str, Any]],
    *,
    warnings: list[str],
    summary: Dict[str, Any],
) -> None:
    if payload is None:
        warnings.append("hybrid_blind_gate report missing or unreadable")
        return
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        warnings.append("hybrid_blind_gate.metrics missing or invalid")
        return
    hybrid_accuracy = _as_float(metrics.get("hybrid_accuracy"))
    hybrid_gain = _as_float(metrics.get("hybrid_gain_vs_graph2d"))
    if hybrid_accuracy is None:
        warnings.append(
            "hybrid_blind_gate.metrics.hybrid_accuracy missing or non-numeric"
        )
    if hybrid_gain is None:
        warnings.append(
            "hybrid_blind_gate.metrics.hybrid_gain_vs_graph2d missing or non-numeric"
        )
    summary["gate_hybrid_accuracy"] = hybrid_accuracy
    summary["gate_hybrid_gain_vs_graph2d"] = hybrid_gain


def _validate_calibration_payload(
    payload: Optional[Dict[str, Any]],
    *,
    warnings: list[str],
    summary: Dict[str, Any],
) -> None:
    if payload is None:
        warnings.append("hybrid calibration report missing or unreadable")
        return
    metrics_after = payload.get("metrics_after")
    if not isinstance(metrics_after, dict):
        warnings.append("hybrid calibration metrics_after missing or invalid")
        return
    ece = _as_float(metrics_after.get("ece"))
    if ece is None:
        warnings.append("hybrid calibration metrics_after.ece missing or non-numeric")
    summary["calibration_ece"] = ece


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Hybrid superpass report json and related gate/calibration payloads."
    )
    parser.add_argument(
        "--superpass-json", required=True, help="Path to superpass report json."
    )
    parser.add_argument(
        "--hybrid-blind-gate-report",
        default="",
        help="Optional hybrid blind gate report json.",
    )
    parser.add_argument(
        "--hybrid-calibration-json",
        default="",
        help="Optional hybrid calibration report json.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for validation result json.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failure (exit 1).",
    )
    parser.add_argument(
        "--schema-mode",
        choices=("off", "builtin"),
        default="builtin",
        help="JSON schema validation mode for report payloads.",
    )
    return parser


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    summary: Dict[str, Any] = {}

    superpass_path = Path(str(args.superpass_json)).expanduser()
    superpass_payload = _read_json_object(
        superpass_path, label="superpass", errors=errors
    )
    _validate_with_json_schema(
        payload=superpass_payload,
        label="superpass",
        schema=SUPERPASS_JSON_SCHEMA,
        schema_mode=str(args.schema_mode),
        errors=errors,
        warnings=warnings,
    )
    if superpass_payload is not None:
        _validate_superpass_payload(
            superpass_payload, errors=errors, warnings=warnings, summary=summary
        )

    gate_payload: Optional[Dict[str, Any]] = None
    gate_path_text = str(args.hybrid_blind_gate_report or "").strip()
    if gate_path_text:
        gate_payload = _read_json_object(
            Path(gate_path_text).expanduser(),
            label="hybrid_blind_gate",
            errors=errors,
        )
    _validate_with_json_schema(
        payload=gate_payload,
        label="hybrid_blind_gate",
        schema=HYBRID_BLIND_GATE_JSON_SCHEMA,
        schema_mode=str(args.schema_mode),
        errors=errors,
        warnings=warnings,
    )
    _validate_hybrid_blind_gate_payload(
        payload=gate_payload, warnings=warnings, summary=summary
    )

    calibration_payload: Optional[Dict[str, Any]] = None
    calibration_path_text = str(args.hybrid_calibration_json or "").strip()
    if calibration_path_text:
        calibration_payload = _read_json_object(
            Path(calibration_path_text).expanduser(),
            label="hybrid_calibration",
            errors=errors,
        )
    _validate_with_json_schema(
        payload=calibration_payload,
        label="hybrid_calibration",
        schema=HYBRID_CALIBRATION_JSON_SCHEMA,
        schema_mode=str(args.schema_mode),
        errors=errors,
        warnings=warnings,
    )
    _validate_calibration_payload(
        payload=calibration_payload,
        warnings=warnings,
        summary=summary,
    )

    if errors:
        status = "error"
    elif warnings:
        status = "warn"
    else:
        status = "ok"

    if errors:
        exit_code = 1
    elif warnings and bool(args.strict):
        exit_code = 1
    else:
        exit_code = 0

    output_payload: Dict[str, Any] = {
        "status": status,
        "strict": bool(args.strict),
        "schema_mode": str(args.schema_mode),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "inputs": {
            "superpass_json": str(superpass_path),
            "hybrid_blind_gate_report": gate_path_text,
            "hybrid_calibration_json": calibration_path_text,
        },
        "overall_exit_code": exit_code,
    }

    output_path_text = str(args.output_json or "").strip()
    if output_path_text:
        _write_json(Path(output_path_text).expanduser(), output_payload)

    print(json.dumps(output_payload, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
