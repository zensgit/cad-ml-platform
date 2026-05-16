#!/usr/bin/env python3
"""Validate the STEP/IGES B-Rep golden manifest contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA_VERSION = "brep_golden_manifest.v1"
ALLOWED_FORMATS = {"step", "stp", "iges", "igs"}
ALLOWED_SOURCE_TYPES = {
    "real_world",
    "vendor",
    "public_cad",
    "internal",
    "fixture",
    "synthetic_demo",
    "generated_mock",
}
RELEASE_EXCLUDED_SOURCE_TYPES = {"fixture", "synthetic_demo", "generated_mock"}
ALLOWED_EXPECTED_BEHAVIORS = {"parse_success", "parse_failure", "graph_failure"}
DEFAULT_MIN_RELEASE_SAMPLES = 50


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    return payload


def _case_format(path_text: str, explicit_format: str = "") -> str:
    if explicit_format:
        return explicit_format.strip().lower()
    return Path(path_text).suffix.lower().lstrip(".")


def _resolve_manifest_root(manifest: Dict[str, Any], manifest_path: Optional[Path]) -> Path:
    raw_root = str(manifest.get("root") or ".")
    root = Path(raw_root).expanduser()
    if root.is_absolute():
        return root.resolve()
    base = manifest_path.parent if manifest_path else Path.cwd()
    return (base / root).resolve()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _require_string(
    case: Dict[str, Any],
    key: str,
    errors: List[str],
    *,
    case_label: str,
) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{case_label}: missing required string field `{key}`")
        return ""
    return value.strip()


def _validate_expected_topology(
    *,
    case: Dict[str, Any],
    case_label: str,
    errors: List[str],
) -> None:
    topology = case.get("expected_topology")
    if not isinstance(topology, dict):
        errors.append(f"{case_label}: parse_success requires `expected_topology` object")
        return
    for key in ("faces_min", "edges_min", "solids_min", "graph_nodes_min"):
        value = topology.get(key)
        if not isinstance(value, int) or value < 0:
            errors.append(f"{case_label}: `expected_topology.{key}` must be an integer >= 0")
    surface_types = topology.get("surface_types")
    if surface_types is not None:
        if not isinstance(surface_types, list) or not all(
            isinstance(item, str) and item.strip() for item in surface_types
        ):
            errors.append(f"{case_label}: `expected_topology.surface_types` must be string list")


def _iter_cases(manifest: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def validate_manifest(
    manifest: Dict[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    min_release_samples: int = DEFAULT_MIN_RELEASE_SAMPLES,
    allow_missing_files: bool = False,
) -> Dict[str, Any]:
    """Validate a B-Rep golden manifest and return a report."""
    errors: List[str] = []
    warnings: List[str] = []
    case_ids = set()
    root = _resolve_manifest_root(manifest, manifest_path)
    cases = list(_iter_cases(manifest))
    format_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    release_eligible_count = 0

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"`schema_version` must be `{SCHEMA_VERSION}`")
    if not isinstance(manifest.get("name"), str) or not str(manifest.get("name")).strip():
        errors.append("missing required string field `name`")
    if not cases:
        errors.append("manifest must include at least one case")

    for index, case in enumerate(cases):
        case_id = _require_string(case, "id", errors, case_label=f"case[{index}]")
        case_label = f"case `{case_id}`" if case_id else f"case[{index}]"
        if case_id:
            if case_id in case_ids:
                errors.append(f"{case_label}: duplicate case id")
            case_ids.add(case_id)

        path_text = _require_string(case, "path", errors, case_label=case_label)
        source_type = _require_string(case, "source_type", errors, case_label=case_label)
        expected_behavior = _require_string(
            case,
            "expected_behavior",
            errors,
            case_label=case_label,
        )
        _require_string(case, "part_family", errors, case_label=case_label)
        _require_string(case, "license", errors, case_label=case_label)

        file_format = _case_format(path_text, str(case.get("format") or ""))
        format_counts[file_format] += 1
        source_type_counts[source_type] += 1
        behavior_counts[expected_behavior] += 1

        if file_format not in ALLOWED_FORMATS:
            errors.append(f"{case_label}: unsupported file format `{file_format}`")
        if source_type not in ALLOWED_SOURCE_TYPES:
            errors.append(f"{case_label}: unsupported source_type `{source_type}`")
        if expected_behavior not in ALLOWED_EXPECTED_BEHAVIORS:
            errors.append(f"{case_label}: unsupported expected_behavior `{expected_behavior}`")

        if path_text:
            file_path = Path(path_text).expanduser()
            if not file_path.is_absolute():
                file_path = root / file_path
            if not allow_missing_files and not file_path.exists():
                errors.append(f"{case_label}: file not found `{path_text}`")

        if expected_behavior == "parse_success":
            _validate_expected_topology(case=case, case_label=case_label, errors=errors)
        elif expected_behavior in {"parse_failure", "graph_failure"}:
            if not isinstance(case.get("expected_failure_reason"), str) or not str(
                case.get("expected_failure_reason")
            ).strip():
                errors.append(
                    f"{case_label}: {expected_behavior} requires `expected_failure_reason`"
                )

        inferred_release_eligible = (
            source_type not in RELEASE_EXCLUDED_SOURCE_TYPES
            and expected_behavior == "parse_success"
        )
        release_eligible = _as_bool(
            case.get("release_eligible"),
            default=inferred_release_eligible,
        )
        if source_type in RELEASE_EXCLUDED_SOURCE_TYPES and release_eligible:
            errors.append(
                f"{case_label}: `{source_type}` cases cannot be release_eligible"
            )
        if release_eligible and expected_behavior != "parse_success":
            errors.append(f"{case_label}: release_eligible requires parse_success")
        if release_eligible:
            release_eligible_count += 1

    if release_eligible_count < min_release_samples:
        warnings.append(
            "release_eligible_count below minimum: "
            f"{release_eligible_count} < {min_release_samples}"
        )

    if errors:
        status = "invalid"
    elif release_eligible_count >= min_release_samples:
        status = "release_ready"
    else:
        status = "insufficient_release_samples"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ready_for_release": status == "release_ready",
        "manifest_path": str(manifest_path) if manifest_path else "",
        "root": str(root),
        "case_count": len(cases),
        "release_eligible_count": release_eligible_count,
        "min_release_samples": min_release_samples,
        "format_counts": dict(format_counts),
        "source_type_counts": dict(source_type_counts),
        "expected_behavior_counts": dict(behavior_counts),
        "errors": errors,
        "warnings": warnings,
    }


def _write_report(path_text: str, report: Dict[str, Any]) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--min-release-samples", type=int, default=DEFAULT_MIN_RELEASE_SAMPLES)
    parser.add_argument("--allow-missing-files", action="store_true")
    parser.add_argument(
        "--fail-on-not-release-ready",
        action="store_true",
        help="Exit non-zero when the manifest is invalid or below the release sample floor.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser()
    manifest = _load_json(manifest_path)
    report = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        min_release_samples=args.min_release_samples,
        allow_missing_files=args.allow_missing_files,
    )
    _write_report(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "invalid":
        return 1
    if args.fail_on_not_release_ready and report["status"] != "release_ready":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
