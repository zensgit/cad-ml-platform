from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_brep_golden_manifest import validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validate_brep_golden_manifest.py"


def _write_case_file(root: Path, name: str = "part.step") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
    return path


def _case(case_id: str, path: str = "part.step") -> dict:
    """Canonical release-eligible case: human-`verified` topology with full
    license/topology provenance, so it counts toward both the sample floor and
    the verified-topology floor."""
    return {
        "id": case_id,
        "path": path,
        "format": "step",
        "source_type": "real_world",
        "release_eligible": True,
        "part_family": "block",
        "license": "internal",
        "license_status": "internal",
        "license_source": "internal-dataset://golden/v1",
        "topology_source": "verified",
        "topology_evidence": "manual-review://golden/v1",
        "expected_behavior": "parse_success",
        "expected_topology": {
            "faces_min": 1,
            "edges_min": 0,
            "solids_min": 0,
            "graph_nodes_min": 1,
            "surface_types": ["plane"],
        },
    }


def _derived_case(case_id: str, path: str = "part.step") -> dict:
    """Release-eligible case whose expected topology was parser-`derived` (not
    human-verified). Counts toward the sample floor but NOT the verified floor;
    `derived` needs no evidence pointer."""
    case = _case(case_id, path)
    case["topology_source"] = "derived"
    case.pop("topology_evidence", None)
    return case


def test_validate_manifest_reports_release_ready_with_enough_real_cases(tmp_path: Path) -> None:
    cases = []
    for index in range(50):
        rel_path = f"parts/part_{index}.step"
        _write_case_file(tmp_path, rel_path)
        cases.append(_case(f"part_{index}", rel_path))
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "release manifest",
        "root": str(tmp_path),
        "cases": cases,
    }

    report = validate_manifest(manifest, min_release_samples=50)

    assert report["status"] == "release_ready"
    assert report["ready_for_release"] is True
    assert report["release_eligible_count"] == 50
    assert report["errors"] == []


def test_validate_manifest_excludes_fixture_from_release_count(tmp_path: Path) -> None:
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "fixture manifest",
        "root": str(tmp_path),
        "cases": [
            {
                **_case("fixture_part"),
                "source_type": "fixture",
                "release_eligible": False,
            }
        ],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "insufficient_release_samples"
    assert report["release_eligible_count"] == 0
    assert report["warnings"]


def test_validate_manifest_rejects_fixture_marked_release_eligible(tmp_path: Path) -> None:
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "bad fixture manifest",
        "root": str(tmp_path),
        "cases": [{**_case("fixture_part"), "source_type": "fixture"}],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("cannot be release_eligible" in error for error in report["errors"])


def test_public_nc_excluded_from_release_count(tmp_path: Path) -> None:
    """ABC-style NonCommercial public CAD: allowed for coverage, never
    counted toward the release floor (Stage 2a decision 2026-05-18)."""
    cases = []
    # 50 internal release-eligible parts.
    for index in range(50):
        rel = f"internal/part_{index}.step"
        _write_case_file(tmp_path, rel)
        cases.append(_case(f"internal_{index}", rel))
    # 10 public_nc coverage parts (must NOT lift release count beyond 50).
    for index in range(10):
        rel = f"abc/abc_{index}.step"
        _write_case_file(tmp_path, rel)
        cases.append(
            {
                **_case(f"abc_{index}", rel),
                "source_type": "public_nc",
                "release_eligible": False,
                "license": "CC-BY-NC-SA-4.0",
                "license_status": "non_commercial",
            }
        )
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "internal + public_nc coverage manifest",
        "root": str(tmp_path),
        "cases": cases,
    }

    report = validate_manifest(manifest, min_release_samples=50)

    assert report["status"] == "release_ready"
    # public_nc parts present but excluded — count stays at the 50 internal.
    assert report["release_eligible_count"] == 50
    assert report["case_count"] == 60
    assert report["source_type_counts"].get("public_nc") == 10
    assert report["errors"] == []


def test_public_nc_marked_release_eligible_is_rejected(tmp_path: Path) -> None:
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "bad public_nc manifest",
        "root": str(tmp_path),
        "cases": [
            {
                **_case("abc_part"),
                "source_type": "public_nc",
                "release_eligible": True,
                "license": "CC-BY-NC-SA-4.0",
                "license_status": "non_commercial",
            }
        ],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("cannot be release_eligible" in error for error in report["errors"])


def test_validate_manifest_rejects_duplicate_ids_and_missing_files(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "bad manifest",
        "root": str(tmp_path),
        "cases": [_case("dup"), _case("dup")],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("duplicate case id" in error for error in report["errors"])
    assert any("file not found" in error for error in report["errors"])


def test_validate_manifest_requires_failure_reason_for_expected_parse_failure(
    tmp_path: Path,
) -> None:
    _write_case_file(tmp_path, "bad.step")
    bad_case = {
        **_case("bad_part", "bad.step"),
        "release_eligible": False,
        "expected_behavior": "parse_failure",
    }
    bad_case.pop("expected_topology")
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "parse failure manifest",
        "root": str(tmp_path),
        "cases": [bad_case],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("expected_failure_reason" in error for error in report["errors"])


def test_validate_manifest_cli_writes_report_for_example_manifest(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            "config/brep_golden_manifest.example.json",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "insufficient_release_samples"
    assert json.loads(output_json.read_text(encoding="utf-8"))["case_count"] == 1


def test_validate_manifest_cli_can_fail_when_not_release_ready(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            "config/brep_golden_manifest.example.json",
            "--output-json",
            str(output_json),
            "--fail-on-not-release-ready",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(output_json.read_text(encoding="utf-8"))["ready_for_release"] is False


def test_release_eligible_with_todo_tag_is_rejected(tmp_path: Path) -> None:
    """Defense-in-depth: a release_eligible case still carrying a TODO-*
    tag has not been human-reviewed and must invalidate the manifest,
    independent of how it got marked eligible (scaffolder regression or
    hand-edit). Guards the PR #480 blocking finding at the validator."""
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "todo-tag manifest",
        "root": str(tmp_path),
        "cases": [{**_case("todo_part"), "tags": ["TODO-source-type"]}],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("unfilled skeleton placeholders" in e for e in report["errors"])


def test_release_eligible_with_todo_field_value_is_rejected(tmp_path: Path) -> None:
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "todo-field manifest",
        "root": str(tmp_path),
        "cases": [{**_case("todo_part"), "license": "TODO"}],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("unfilled skeleton placeholders" in e for e in report["errors"])


def test_non_release_case_may_keep_todo_placeholders(tmp_path: Path) -> None:
    """The TODO gate only applies to release_eligible cases. A fixture /
    public_nc case (release-excluded) may legitimately retain TODO tags
    without invalidating the manifest."""
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "non-release todo manifest",
        "root": str(tmp_path),
        "cases": [
            {
                **_case("fixture_part"),
                "source_type": "fixture",
                "release_eligible": False,
                "license": "TODO",
                "tags": ["TODO-license", "TODO-topology"],
            }
        ],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    # Below floor (0 eligible) but NOT invalid — TODO on a non-release
    # case is allowed.
    assert report["status"] == "insufficient_release_samples"
    assert report["errors"] == []


def test_release_eligible_with_string_tags_is_rejected(tmp_path: Path) -> None:
    """A hand-written manifest using a STRING tags value
    (`tags: "TODO-source-type"`) must not bypass the TODO gate by
    per-character iteration. It is both a schema violation (tags must
    be a list) and, defensively, still caught by the TODO gate."""
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "string-tags manifest",
        "root": str(tmp_path),
        "cases": [{**_case("todo_part"), "tags": "TODO-source-type"}],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("`tags` must be a list" in e for e in report["errors"])
    assert any("unfilled skeleton placeholders" in e for e in report["errors"])


def test_non_todo_string_tags_still_rejected_as_schema_violation(tmp_path: Path) -> None:
    """Even a benign string tags value is a schema violation (would
    iterate per-character elsewhere); reject it regardless of TODO."""
    _write_case_file(tmp_path)
    manifest = {
        "schema_version": "brep_golden_manifest.v1",
        "name": "benign string-tags manifest",
        "root": str(tmp_path),
        "cases": [{**_case("p"), "tags": "release"}],
    }

    report = validate_manifest(manifest, min_release_samples=1)

    assert report["status"] == "invalid"
    assert any("`tags` must be a list" in e for e in report["errors"])


# --- Stage 2a provenance contract: license_status / topology_source ---


def _single(tmp_path: Path, case: dict, name: str) -> dict:
    _write_case_file(tmp_path)
    return validate_manifest(
        {
            "schema_version": "brep_golden_manifest.v1",
            "name": name,
            "root": str(tmp_path),
            "cases": [case],
        },
        min_release_samples=1,
    )


def test_release_eligible_requires_license_status(tmp_path: Path) -> None:
    case = _case("no_license_status")
    case.pop("license_status")
    report = _single(tmp_path, case, "missing license_status")
    assert report["status"] == "invalid"
    assert any("requires `license_status`" in e for e in report["errors"])


def test_release_usable_license_requires_source(tmp_path: Path) -> None:
    case = _case("no_license_source")
    case.pop("license_source")
    report = _single(tmp_path, case, "missing license_source")
    assert report["status"] == "invalid"
    assert any("requires a non-empty `license_source`" in e for e in report["errors"])


def test_excluded_license_status_cannot_be_release_eligible(tmp_path: Path) -> None:
    case = {**_case("unverified_part"), "license_status": "unverified"}
    report = _single(tmp_path, case, "unverified license")
    assert report["status"] == "invalid"
    assert any("cannot be release_eligible" in e for e in report["errors"])


def test_invalid_license_status_value_is_rejected(tmp_path: Path) -> None:
    case = {**_case("bad_status"), "license_status": "totally-made-up"}
    report = _single(tmp_path, case, "bad license_status")
    assert report["status"] == "invalid"
    assert any("`license_status` must be one of" in e for e in report["errors"])


def test_public_nc_must_be_non_commercial(tmp_path: Path) -> None:
    """The source axis (public_nc) and license axis (non_commercial) describe
    the same NonCommercial fact and must not silently disagree."""
    case = {
        **_case("contradiction"),
        "source_type": "public_nc",
        "release_eligible": False,
        "license_status": "permissive",
    }
    report = _single(tmp_path, case, "contradicting axes")
    assert report["status"] == "invalid"
    assert any("requires license_status `non_commercial`" in e for e in report["errors"])


def test_release_eligible_requires_topology_source(tmp_path: Path) -> None:
    case = _case("no_topo_source")
    case.pop("topology_source")
    report = _single(tmp_path, case, "missing topology_source")
    assert report["status"] == "invalid"
    assert any("requires `topology_source`" in e for e in report["errors"])


def test_verified_topology_requires_evidence(tmp_path: Path) -> None:
    case = _case("verified_no_evidence")
    case.pop("topology_evidence")
    report = _single(tmp_path, case, "verified without evidence")
    assert report["status"] == "invalid"
    assert any("requires non-empty `topology_evidence`" in e for e in report["errors"])


def test_verified_topology_requires_positive_faces_min(tmp_path: Path) -> None:
    case = _case("verified_zero_faces")
    case["expected_topology"] = {**case["expected_topology"], "faces_min": 0}
    report = _single(tmp_path, case, "verified zero faces_min")
    assert report["status"] == "invalid"
    assert any("faces_min` > 0" in e for e in report["errors"])


def test_derived_case_needs_no_evidence(tmp_path: Path) -> None:
    """A `derived` case is valid without topology_evidence (it makes no
    human-verified claim)."""
    report = _single(tmp_path, _derived_case("derived_ok"), "single derived case")
    # 1 eligible derived: below the (min 10) verified floor, but not INVALID.
    assert report["status"] == "insufficient_verified_topology"
    assert report["errors"] == []
    assert report["derived_topology_count"] == 1


def test_derived_only_set_blocks_release_below_verified_floor(tmp_path: Path) -> None:
    """50 release-eligible cases all parser-`derived`: meets the sample floor
    but not the verified floor, so the manifest is NOT release_ready. Binding
    gate against a tautological golden set."""
    cases = []
    for index in range(50):
        rel = f"derived/part_{index}.step"
        _write_case_file(tmp_path, rel)
        cases.append(_derived_case(f"derived_{index}", rel))
    report = validate_manifest(
        {
            "schema_version": "brep_golden_manifest.v1",
            "name": "all-derived manifest",
            "root": str(tmp_path),
            "cases": cases,
        },
        min_release_samples=50,
    )
    assert report["status"] == "insufficient_verified_topology"
    assert report["ready_for_release"] is False
    assert report["release_eligible_count"] == 50
    assert report["verified_topology_count"] == 0
    assert report["derived_topology_count"] == 50
    assert report["verified_topology_floor"] == 10
    assert report["verified_topology_floor_met"] is False
    assert any("topology_verified_below_release_floor" in w for w in report["warnings"])
    assert report["errors"] == []


def test_verified_floor_boundary(tmp_path: Path) -> None:
    """At 50 eligible the floor is max(10, ceil(0.2*50)) = 10: 10 verified +
    40 derived is release_ready; 9 verified + 41 derived is not."""

    def _build(num_verified: int) -> dict:
        cases = []
        for index in range(num_verified):
            rel = f"v/part_{index}.step"
            _write_case_file(tmp_path, rel)
            cases.append(_case(f"v_{index}", rel))
        for index in range(50 - num_verified):
            rel = f"d/part_{index}.step"
            _write_case_file(tmp_path, rel)
            cases.append(_derived_case(f"d_{index}", rel))
        return {
            "schema_version": "brep_golden_manifest.v1",
            "name": f"{num_verified}-verified manifest",
            "root": str(tmp_path),
            "cases": cases,
        }

    ok = validate_manifest(_build(10), min_release_samples=50)
    assert ok["status"] == "release_ready"
    assert ok["verified_topology_floor"] == 10
    assert ok["verified_topology_count"] == 10

    short = validate_manifest(_build(9), min_release_samples=50)
    assert short["status"] == "insufficient_verified_topology"
    assert short["ready_for_release"] is False


def test_report_exposes_provenance_layered_counts(tmp_path: Path) -> None:
    cases = []
    for index in range(50):
        rel = f"v/part_{index}.step"
        _write_case_file(tmp_path, rel)
        cases.append(_case(f"v_{index}", rel))
    report = validate_manifest(
        {
            "schema_version": "brep_golden_manifest.v1",
            "name": "provenance counts manifest",
            "root": str(tmp_path),
            "cases": cases,
        },
        min_release_samples=50,
    )
    assert report["status"] == "release_ready"
    assert report["verified_topology_count"] == 50
    assert report["derived_topology_count"] == 0
    assert report["license_status_counts"].get("internal") == 50
    assert report["verified_topology_floor_met"] is True
