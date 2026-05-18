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
    return {
        "id": case_id,
        "path": path,
        "format": "step",
        "source_type": "real_world",
        "release_eligible": True,
        "part_family": "block",
        "license": "internal",
        "expected_behavior": "parse_success",
        "expected_topology": {
            "faces_min": 1,
            "edges_min": 0,
            "solids_min": 0,
            "graph_nodes_min": 1,
            "surface_types": ["plane"],
        },
    }


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
