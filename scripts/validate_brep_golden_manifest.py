#!/usr/bin/env python3
"""Validate the STEP/IGES B-Rep golden manifest contract."""

from __future__ import annotations

import argparse
import json
import math
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
    "public_nc",
    "fixture",
    "synthetic_demo",
    "generated_mock",
}
# `public_nc`: real public CAD whose license carries a NonCommercial clause
# (e.g. ABC Dataset, CC-BY-NC-SA 4.0). Usable for parser/topology coverage
# but MUST NOT count toward the release floor — shipping NC data as a
# release-gating benchmark in a commercial product is a licensing risk.
RELEASE_EXCLUDED_SOURCE_TYPES = {
    "fixture",
    "synthetic_demo",
    "generated_mock",
    "public_nc",
}
ALLOWED_EXPECTED_BEHAVIORS = {"parse_success", "parse_failure", "graph_failure"}
DEFAULT_MIN_RELEASE_SAMPLES = 50

# --- Stage 2a provenance contract (license_status / topology_source) ---
# These fields are OPTIONAL on the schema (so non-release fixture/NC/failure
# rows and the existing example manifest stay valid) but REQUIRED on any case
# that counts toward the release floor. The validator cannot verify that a
# license or topology claim is *true*; it enforces that the claim is present,
# well-formed, and carries an auditable pointer — provenance capture, not
# verification. The point is to make a false "green" attributable, never silent.
ALLOWED_LICENSE_STATUSES = {
    "internal",
    "public_domain",
    "permissive",
    "proprietary_authorized",
    "non_commercial",
    "unverified",
}
# License statuses under which a case MAY count toward the release floor. Each
# requires a non-empty `license_source` so the (unverifiable) claim is at least
# attributable to a named origin / authorization record.
RELEASE_USABLE_LICENSE_STATUSES = {
    "internal",
    "public_domain",
    "permissive",
    "proprietary_authorized",
}
# License statuses that force release-ineligibility (NonCommercial or unproven).
RELEASE_EXCLUDED_LICENSE_STATUSES = {"non_commercial", "unverified"}
ALLOWED_TOPOLOGY_SOURCES = {"verified", "derived"}
# Verified-topology release floor: at least this many release-eligible cases
# must carry human-`verified` (not parser-`derived`) expected topology, so the
# golden set is not validated tautologically by the same parser under test.
RELEASE_VERIFIED_TOPOLOGY_FLOOR_MIN = 10
RELEASE_VERIFIED_TOPOLOGY_FLOOR_RATIO = 0.2


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


def _verified_topology_floor(release_eligible_count: int) -> int:
    """Minimum human-verified topologies required at a given release-floor size.

    `max(MIN, ceil(RATIO * N))`, capped at N: the floor can never demand more
    verified cases than exist. The cap binds only for N < MIN (tiny/test
    manifests); for any production floor (N >= 50) the cap never binds and the
    value equals the spec `max(10, ceil(0.2 * N))`.
    """
    target = max(
        RELEASE_VERIFIED_TOPOLOGY_FLOOR_MIN,
        math.ceil(RELEASE_VERIFIED_TOPOLOGY_FLOOR_RATIO * release_eligible_count),
    )
    return min(target, release_eligible_count)


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
    license_status_counts: Counter[str] = Counter()
    release_eligible_count = 0
    verified_topology_count = 0
    derived_topology_count = 0

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

        raw_tags = case.get("tags")
        if raw_tags is not None and not isinstance(raw_tags, list):
            # A string `tags` value iterates per-character, which would let
            # `tags: "TODO-source-type"` slip past the TODO release gate.
            # Reject the schema violation outright.
            errors.append(
                f"{case_label}: `tags` must be a list of strings, got "
                f"{type(raw_tags).__name__}"
            )

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

        # --- license / topology provenance (Stage 2a) ---
        # Validate the controlled vocabulary whenever a field is present (a typo
        # is a schema error regardless of eligibility); the fields are only
        # *required* on release_eligible cases, in the block below.
        license_status_norm = ""
        raw_license_status = case.get("license_status")
        if raw_license_status is not None:
            if (
                not isinstance(raw_license_status, str)
                or raw_license_status.strip().lower() not in ALLOWED_LICENSE_STATUSES
            ):
                errors.append(
                    f"{case_label}: `license_status` must be one of "
                    f"{sorted(ALLOWED_LICENSE_STATUSES)}"
                )
            else:
                license_status_norm = raw_license_status.strip().lower()
                license_status_counts[license_status_norm] += 1

        topology_source_norm = ""
        raw_topology_source = case.get("topology_source")
        if raw_topology_source is not None:
            if (
                not isinstance(raw_topology_source, str)
                or raw_topology_source.strip().lower() not in ALLOWED_TOPOLOGY_SOURCES
            ):
                errors.append(
                    f"{case_label}: `topology_source` must be one of "
                    f"{sorted(ALLOWED_TOPOLOGY_SOURCES)}"
                )
            else:
                topology_source_norm = raw_topology_source.strip().lower()

        # `public_nc` (source axis) and `non_commercial` (license axis) are the
        # same NonCommercial fact, so the two exclusion axes must agree BOTH
        # ways: neither may mark a case excluded while the other leaves it
        # release-usable. Enforce the full biconditional public_nc <-> non_commercial.
        if (
            source_type == "public_nc"
            and license_status_norm
            and license_status_norm != "non_commercial"
        ):
            errors.append(
                f"{case_label}: source_type `public_nc` requires license_status "
                f"`non_commercial` (got `{license_status_norm}`)"
            )
        if (
            license_status_norm == "non_commercial"
            and source_type
            and source_type != "public_nc"
        ):
            errors.append(
                f"{case_label}: license_status `non_commercial` requires source_type "
                f"`public_nc` (got `{source_type}`)"
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
            # Defense-in-depth: a release_eligible case carrying unfilled
            # skeleton placeholders (TODO field values or TODO-* tags) has
            # not been human-reviewed and must not pass the release gate,
            # even if the scaffolder or a hand-edit left it eligible. This
            # is independent of the scaffolder so a regression there (or a
            # hand-written manifest) cannot bypass it.
            # Normalize defensively: a str `tags` (already a schema error
            # above) would iterate per-character and bypass this gate, so
            # treat a bare string as a single tag here too — the gate must
            # hold even if the schema check is ever relaxed.
            raw_tags_value = case.get("tags")
            if isinstance(raw_tags_value, list):
                tag_iter = raw_tags_value
            elif isinstance(raw_tags_value, str):
                tag_iter = [raw_tags_value]
            else:
                tag_iter = []
            todo_tags = [
                str(tag) for tag in tag_iter if str(tag).startswith("TODO-")
            ]
            todo_fields = [
                field
                for field in (
                    "part_family",
                    "license",
                    "license_source",
                    "topology_evidence",
                )
                if str(case.get(field) or "").strip().upper() == "TODO"
            ]
            if todo_tags or todo_fields:
                errors.append(
                    f"{case_label}: release_eligible case still has unfilled "
                    f"skeleton placeholders "
                    f"(tags={todo_tags or '[]'}, fields={todo_fields or '[]'}); "
                    f"a human must fill them and clear the TODO markers "
                    f"before it can count toward the release floor"
                )

            # Provenance is mandatory for anything counting toward the release
            # floor: an auditable license classification (+ source) and a
            # declared topology origin. Without these the floor is a bare row
            # count that says nothing about trustworthiness.
            if not license_status_norm:
                errors.append(
                    f"{case_label}: release_eligible requires `license_status` "
                    f"(one of {sorted(RELEASE_USABLE_LICENSE_STATUSES)})"
                )
            elif license_status_norm in RELEASE_EXCLUDED_LICENSE_STATUSES:
                errors.append(
                    f"{case_label}: license_status `{license_status_norm}` cannot be "
                    f"release_eligible"
                )
            elif (
                license_status_norm in RELEASE_USABLE_LICENSE_STATUSES
                and not str(case.get("license_source") or "").strip()
            ):
                errors.append(
                    f"{case_label}: license_status `{license_status_norm}` requires a "
                    f"non-empty `license_source` (auditable provenance pointer)"
                )

            if not topology_source_norm:
                errors.append(
                    f"{case_label}: release_eligible requires `topology_source` "
                    f"(`verified` or `derived`)"
                )
            elif topology_source_norm == "verified":
                # `verified` must have teeth or the verified floor is gameable by
                # relabeling: require an evidence pointer and a non-trivial face
                # floor. faces_min only — surface-only parts legitimately have
                # solids_min == 0.
                if not str(case.get("topology_evidence") or "").strip():
                    errors.append(
                        f"{case_label}: topology_source `verified` requires non-empty "
                        f"`topology_evidence`"
                    )
                topo = case.get("expected_topology")
                faces_min = topo.get("faces_min") if isinstance(topo, dict) else None
                if not isinstance(faces_min, int) or faces_min <= 0:
                    errors.append(
                        f"{case_label}: topology_source `verified` requires "
                        f"`expected_topology.faces_min` > 0"
                    )

            if topology_source_norm == "verified":
                verified_topology_count += 1
            elif topology_source_norm == "derived":
                derived_topology_count += 1
            release_eligible_count += 1

    verified_topology_floor = _verified_topology_floor(release_eligible_count)
    verified_topology_floor_met = verified_topology_count >= verified_topology_floor

    if release_eligible_count < min_release_samples:
        warnings.append(
            "release_eligible_count below minimum: "
            f"{release_eligible_count} < {min_release_samples}"
        )
    elif not verified_topology_floor_met:
        # Sample floor met, but too few human-verified topologies: the set is
        # carried by `derived` rows whose expected_topology came from the same
        # parser under test (tautological). Block release and emit the named
        # flag the forward scorecard consumes.
        warnings.append(
            "topology_verified_below_release_floor: "
            f"{verified_topology_count} < {verified_topology_floor} verified "
            f"(release_eligible={release_eligible_count})"
        )

    if errors:
        status = "invalid"
    elif release_eligible_count < min_release_samples:
        status = "insufficient_release_samples"
    elif not verified_topology_floor_met:
        status = "insufficient_verified_topology"
    else:
        status = "release_ready"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "ready_for_release": status == "release_ready",
        "manifest_path": str(manifest_path) if manifest_path else "",
        "root": str(root),
        "case_count": len(cases),
        "release_eligible_count": release_eligible_count,
        "min_release_samples": min_release_samples,
        "verified_topology_count": verified_topology_count,
        "derived_topology_count": derived_topology_count,
        "verified_topology_floor": verified_topology_floor,
        "verified_topology_floor_met": verified_topology_floor_met,
        "format_counts": dict(format_counts),
        "source_type_counts": dict(source_type_counts),
        "expected_behavior_counts": dict(behavior_counts),
        "license_status_counts": dict(license_status_counts),
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
