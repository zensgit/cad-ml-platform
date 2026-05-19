"""Tests for the brep golden manifest skeleton scaffolder.

OCC-free by design (matches the scaffolder's no-topology-probe contract).
The skeleton it emits must round-trip through the real validator with
`--allow-missing-files`-equivalent semantics (we write real placeholder
files so paths resolve).
"""

from __future__ import annotations

from pathlib import Path

from scripts.build_brep_golden_manifest_skeleton import (
    SCHEMA_VERSION,
    build_skeleton,
    summarize,
)
from scripts.validate_brep_golden_manifest import validate_manifest


def _touch_step(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")


def test_skeleton_infers_source_type_from_first_path_segment(tmp_path: Path) -> None:
    _touch_step(tmp_path, "internal/bracket_a.step")
    _touch_step(tmp_path, "public_nc/abc_00001.step")
    _touch_step(tmp_path, "vendor/acme/shaft.stp")

    manifest = build_skeleton(tmp_path)

    assert manifest["schema_version"] == SCHEMA_VERSION
    by_id = {c["id"]: c for c in manifest["cases"]}
    assert by_id["internal_bracket_a"]["source_type"] == "internal"
    assert by_id["internal_bracket_a"]["release_eligible"] is True
    assert by_id["public_nc_abc_00001"]["source_type"] == "public_nc"
    # public_nc is release-excluded -> must NOT be release_eligible.
    assert by_id["public_nc_abc_00001"]["release_eligible"] is False
    assert by_id["vendor_acme_shaft"]["source_type"] == "vendor"
    assert by_id["vendor_acme_shaft"]["format"] == "stp"


def test_unknown_bucket_defaults_internal_but_flagged(tmp_path: Path) -> None:
    _touch_step(tmp_path, "weird_folder/thing.iges")

    manifest = build_skeleton(tmp_path)
    case = manifest["cases"][0]

    assert case["source_type"] == "internal"
    # Must be tagged so it cannot silently enter the release floor
    # mis-classified.
    assert "TODO-source-type" in case["tags"]
    assert case["format"] == "iges"


def test_todo_fields_are_explicit_placeholders(tmp_path: Path) -> None:
    _touch_step(tmp_path, "internal/p1.step")

    case = build_skeleton(tmp_path)["cases"][0]

    assert case["part_family"] == "TODO"
    assert case["license"] == "TODO"
    assert "TODO-part-family" in case["tags"]
    assert "TODO-license" in case["tags"]
    assert "TODO-topology" in case["tags"]
    # Conservative topology placeholder (validator-valid, not real minima).
    assert case["expected_topology"]["faces_min"] == 1
    assert case["expected_topology"]["surface_types"] == []


def test_duplicate_stems_get_unique_ids(tmp_path: Path) -> None:
    _touch_step(tmp_path, "internal/a/part.step")
    _touch_step(tmp_path, "internal/b/part.step")

    ids = [c["id"] for c in build_skeleton(tmp_path)["cases"]]

    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_empty_root_yields_zero_cases(tmp_path: Path) -> None:
    manifest = build_skeleton(tmp_path)
    assert manifest["cases"] == []
    assert summarize(manifest)["case_count"] == 0


def test_summary_counts_release_eligible_and_todo_source(tmp_path: Path) -> None:
    for i in range(3):
        _touch_step(tmp_path, f"internal/p{i}.step")
    _touch_step(tmp_path, "public_nc/abc.step")
    _touch_step(tmp_path, "mystery/x.step")

    summary = summarize(build_skeleton(tmp_path))

    assert summary["case_count"] == 5
    # 3 internal eligible + mystery(defaults internal, eligible) = 4;
    # public_nc excluded.
    assert summary["release_eligible_count"] == 4
    assert summary["source_type_counts"]["public_nc"] == 1
    assert summary["needs_source_type_review"] == 1


def test_skeleton_is_structurally_valid_against_real_validator(tmp_path: Path) -> None:
    """A skeleton with TODO fields filled in must pass the real validator.

    We simulate the human filling step minimally: set part_family /
    license, keep paths real. This proves the scaffolder emits a
    schema-correct shell, not just plausible-looking JSON.
    """
    for i in range(50):
        _touch_step(tmp_path, f"internal/part_{i}.step")
    _touch_step(tmp_path, "public_nc/abc_0.step")

    manifest = build_skeleton(tmp_path, manifest_root=str(tmp_path))
    for case in manifest["cases"]:
        case["part_family"] = "block"
        case["license"] = (
            "CC-BY-NC-SA-4.0" if case["source_type"] == "public_nc" else "internal"
        )

    report = validate_manifest(manifest, min_release_samples=50)

    assert report["status"] == "release_ready", report["errors"]
    # 50 internal eligible; public_nc excluded.
    assert report["release_eligible_count"] == 50
    assert report["case_count"] == 51
