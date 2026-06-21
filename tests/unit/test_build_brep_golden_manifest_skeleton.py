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


def test_known_bucket_only_recognised_in_first_segment(tmp_path: Path) -> None:
    """A known bucket name appearing BELOW the first path segment must
    NOT be picked up — only rel_parts[0] is authoritative (DESIGN §2.2).
    Guards against silent mis-classification: mystery/public_nc/x.step
    must default to internal+flagged, never become public_nc.
    """
    _touch_step(tmp_path, "mystery/public_nc/x.step")
    _touch_step(tmp_path, "mystery/internal/y.step")

    by_id = {c["id"]: c for c in build_skeleton(tmp_path)["cases"]}

    nc_nested = by_id["mystery_public_nc_x"]
    # Deep `public_nc` segment ignored -> defaults to internal AND flagged.
    assert nc_nested["source_type"] == "internal"
    assert "TODO-source-type" in nc_nested["tags"]
    # HARD GATE (not advisory): an un-cleanly-inferred case must NOT be
    # release_eligible. The tag alone was insufficient — the validator
    # only reads release_eligible — so the scaffolder itself must refuse
    # to mark a guessed source as eligible. release_eligible_count for
    # 50 such files must be 0, not 50.
    assert nc_nested["release_eligible"] is False

    internal_nested = by_id["mystery_internal_y"]
    # Even though the deep segment is `internal` and the default is also
    # `internal`, it is a DEFAULT not a clean inference: flagged AND
    # release_eligible=False. Not enforcing this would let a future
    # relayout (mystery/ -> public_nc/) silently flip release-eligibility.
    assert internal_nested["source_type"] == "internal"
    assert "TODO-source-type" in internal_nested["tags"]
    assert internal_nested["release_eligible"] is False


def test_root_level_file_defaults_internal_and_flagged(tmp_path: Path) -> None:
    """A file directly under root (no subdir, empty rel_parts) cannot be
    cleanly inferred — must default internal AND be flagged."""
    _touch_step(tmp_path, "loose.step")

    case = build_skeleton(tmp_path)["cases"][0]

    assert case["source_type"] == "internal"
    assert "TODO-source-type" in case["tags"]
    # Root-level (empty rel_parts) is never a clean inference -> must not
    # be release_eligible on a guess.
    assert case["release_eligible"] is False


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
    # Only the 3 cleanly-inferred internal cases are eligible. public_nc
    # is release-excluded; mystery/ is an un-clean default so it is NOT
    # eligible (hard gate, not advisory tag).
    assert summary["release_eligible_count"] == 3
    assert summary["source_type_counts"]["public_nc"] == 1
    assert summary["needs_source_type_review"] == 1


def test_skeleton_requires_full_human_signoff_to_pass_validator(tmp_path: Path) -> None:
    """The full human-signoff contract: filling field VALUES alone is
    NOT enough — the validator's defense-in-depth rejects any
    release_eligible case that still carries TODO-* tags. A human must
    both fill part_family/license AND clear the TODO markers. This is
    the behavioural contract behind the 'tag forces human confirm
    before it counts' claim (previously false: tags were advisory).
    """
    for i in range(50):
        _touch_step(tmp_path, f"internal/part_{i}.step")
    _touch_step(tmp_path, "public_nc/abc_0.step")

    manifest = build_skeleton(tmp_path, manifest_root=str(tmp_path))

    # Step 1: fill field values only, leave TODO-* tags -> must STILL fail.
    for case in manifest["cases"]:
        case["part_family"] = "block"
        case["license"] = (
            "CC-BY-NC-SA-4.0" if case["source_type"] == "public_nc" else "internal"
        )
    half_filled = validate_manifest(manifest, min_release_samples=50)
    assert half_filled["status"] == "invalid", (
        "fields filled but TODO-* tags retained must NOT pass — the gate "
        "is not advisory"
    )
    assert any(
        "unfilled skeleton placeholders" in e for e in half_filled["errors"]
    )

    # Step 2: full signoff = clear the TODO-* tags AND fill the provenance the
    # Stage 2a contract now requires (license classification + auditable source,
    # and a human-verified topology + evidence) -> now release_ready.
    for case in manifest["cases"]:
        case["tags"] = [t for t in case["tags"] if not t.startswith("TODO-")]
        if case["release_eligible"]:
            case["license_status"] = "internal"
            case["license_source"] = "internal-dataset://golden/v1"
            case["topology_source"] = "verified"
            case["topology_evidence"] = "manual-review://golden/v1"

    report = validate_manifest(manifest, min_release_samples=50)

    assert report["status"] == "release_ready", report["errors"]
    # 50 cleanly-inferred internal eligible; public_nc excluded.
    assert report["release_eligible_count"] == 50
    assert report["case_count"] == 51
    # All 50 eligible are now human-verified -> verified-topology floor met.
    assert report["verified_topology_count"] == 50
    assert report["verified_topology_floor_met"] is True


def test_scaffolder_blocking_finding_regression_50_mystery_public_nc(
    tmp_path: Path,
) -> None:
    """Regression guard for the PR #480 blocking finding: 50
    `mystery/public_nc/*.step` files previously produced 50
    release_eligible cases (release_ready, count=50). They must now
    produce 0 release-eligible cases and the validator must NOT report
    release_ready.
    """
    for i in range(50):
        _touch_step(tmp_path, f"mystery/public_nc/p{i}.step")

    manifest = build_skeleton(tmp_path, manifest_root=str(tmp_path))
    for case in manifest["cases"]:
        case["part_family"] = "block"
        case["license"] = "internal"

    assert all(c["release_eligible"] is False for c in manifest["cases"])
    report = validate_manifest(manifest, min_release_samples=50)
    assert report["status"] != "release_ready"
    assert report["release_eligible_count"] == 0
