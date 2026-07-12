"""Track E slice-2 — versioned manifest (§8.1.6) + real/synthetic/augmented reporting (§8.1.5).

Torch-free: exercises categorize(), the enriched manifest builder (which REUSES slice-1's
leakage-safe split rather than re-deriving it), the category breakdown, and the manifest-level
reproducibility check whose `verify_manifest` goes RED on tamper (content change / added row).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import track_e_manifest as tm


def _mk(tmp_path: Path, name: str, content: bytes, label: str, **extra: str) -> dict:
    p = tmp_path / name
    p.write_bytes(content)
    row = {"file_path": str(p), "cache_path": "", "taxonomy_v2_class": label}
    row.update(extra)
    return row


# --- categorize ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        # review P1: an unmarked, undeclared sample is UNKNOWN provenance, not silently "real"
        ("gear.dxf", "unknown"),
        ("gear_aug1.dxf", "augmented"),
        ("gear_AUG2.DXF", "augmented"),          # case-insensitive
        ("gear_rot90.dxf", "augmented"),
        ("gear_flip.dxf", "augmented"),
        ("gear_flipped.dxf", "augmented"),
        ("gear_noise3.dxf", "augmented"),
        ("gear_jitter.dxf", "augmented"),
        ("gear_scale1.dxf", "augmented"),
        ("part_synth.dxf", "synthetic"),
        ("part_synthetic.dxf", "synthetic"),
        ("PART_GENERATED.dxf", "synthetic"),      # case-insensitive
        ("part_gan.dxf", "synthetic"),
        # adversarial: substrings that must NOT false-positive without a marker boundary → unknown
        ("rotator_part.dxf", "unknown"),
        ("generalpurpose.dxf", "unknown"),
        ("gear2.dxf", "unknown"),                 # bare trailing digit, no marker word
    ],
)
def test_categorize_from_filename(name: str, expected: str) -> None:
    row = {"file_path": name, "cache_path": "", "taxonomy_v2_class": "gear"}
    assert tm.categorize(row) == expected


@pytest.mark.parametrize("declared", ["real", "synthetic", "augmented", "unknown"])
def test_declared_provenance_column_is_authoritative(declared: str) -> None:
    # an explicit data_origin/provenance/category column overrides any filename inference
    row = {"file_path": "gear_aug1.dxf", "taxonomy_v2_class": "gear", "data_origin": declared}
    assert tm.categorize(row) == declared


def test_categorize_checks_declared_family_column_too() -> None:
    aug_row = {"file_path": "weird123.dxf", "family": "gear_augmented", "taxonomy_v2_class": "gear"}
    assert tm.categorize(aug_row) == "augmented"

    synth_row = {"file_path": "weird456.dxf", "family": "gear-synthetic", "taxonomy_v2_class": "gear"}
    assert tm.categorize(synth_row) == "synthetic"


def test_categorize_is_deterministic() -> None:
    row = {"file_path": "part_aug1.dxf", "taxonomy_v2_class": "gear"}
    assert tm.categorize(row) == tm.categorize(dict(row)) == "augmented"


# --- build_versioned_manifest: §8.1.6 fields + quarantine exclusion ------------------------------
def test_build_versioned_manifest_emits_every_8_1_6_field(tmp_path: Path) -> None:
    real_row = _mk(tmp_path, "gear.dxf", b"AAAA", "gear"); real_row["data_origin"] = "real"
    aug_row = _mk(tmp_path, "gear_aug1.dxf", b"BBBB", "gear")
    synth_row = _mk(tmp_path, "part_synth.dxf", b"CCCC", "part")
    plain_row = _mk(tmp_path, "bracket.dxf", b"DDDD", "bracket")  # unmarked, undeclared -> unknown
    missing_row = {
        "file_path": str(tmp_path / "does_not_exist.dxf"),
        "cache_path": "",
        "taxonomy_v2_class": "gear",
    }
    rows = [real_row, aug_row, synth_row, plain_row, missing_row]

    manifest = tm.build_versioned_manifest(
        rows,
        source="acme-archive-2026",
        license_="proprietary-internal",
        label_authority="manifest:taxonomy_v2_class",
    )

    assert manifest["schema_version"] == "evaluation-integrity-manifest-v2"
    assert manifest["source"] == "acme-archive-2026"
    assert manifest["license"] == "proprietary-internal"
    assert manifest["label_authority"] == "manifest:taxonomy_v2_class"
    assert manifest["holdout_fraction"] == tm.DEFAULT_HOLDOUT_FRACTION

    # the unreadable row is EXCLUDED from rows and surfaced in quarantined instead
    assert len(manifest["rows"]) == 4
    file_paths = {r["file_path"] for r in manifest["rows"]}
    assert missing_row["file_path"] not in file_paths
    assert any("unreadable" in q["reason"] for q in manifest["quarantined"])

    required_fields = {
        "file_path",
        "cache_path",
        "taxonomy_v2_class",
        "family",
        "content_hash",
        "split",
        "category",
        "source",
        "license",
        "label_authority",
    }
    for row in manifest["rows"]:
        assert required_fields <= set(row.keys())
        assert row["source"] == "acme-archive-2026"
        assert row["license"] == "proprietary-internal"
        assert row["label_authority"] == "manifest:taxonomy_v2_class"
        assert row["split"] in ("train", "holdout")
        assert row["category"] in tm.CATEGORIES
        assert len(row["content_hash"]) == 64  # sha256 hex

    by_path = {r["file_path"]: r for r in manifest["rows"]}
    assert by_path[real_row["file_path"]]["category"] == "real"          # declared data_origin
    assert by_path[aug_row["file_path"]]["category"] == "augmented"
    assert by_path[synth_row["file_path"]]["category"] == "synthetic"
    assert by_path[plain_row["file_path"]]["category"] == "unknown"      # unmarked -> unknown

    # unknown provenance keeps the manifest not-provenance-complete (review P1)
    assert manifest["unknown_provenance_rows"] == 1
    assert manifest["provenance_complete"] is False

    assert len(manifest["manifest_digest"]) == 64
    assert len(manifest["split_digest"]) == 64


def test_build_versioned_manifest_family_matches_slice1_declared_column(tmp_path: Path) -> None:
    a = _mk(tmp_path, "weird_alpha.dxf", b"1", "gear", family="GEAR-7")
    b = _mk(tmp_path, "totally_other.dxf", b"2", "gear", family="gear-7")
    rows = [a, b] + [_mk(tmp_path, f"{f}.dxf", f.encode(), f) for f in ("m", "n", "o", "p")]
    manifest = tm.build_versioned_manifest(
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.5
    )
    by_path = {r["file_path"]: r for r in manifest["rows"]}
    # same declared family -> same family key -> can never straddle the split (inherited from slice-1)
    assert by_path[a["file_path"]]["family"] == by_path[b["file_path"]]["family"]
    assert by_path[a["file_path"]]["split"] == by_path[b["file_path"]]["split"]


# --- manifest_digest determinism ------------------------------------------------------------------
def test_manifest_digest_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"cls{i % 3}") for i in range(8)]
    m1 = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    m2 = tm.build_versioned_manifest(list(reversed(rows)), source="s", license_="l", label_authority="a")
    assert m1["manifest_digest"] == m2["manifest_digest"]
    assert m1["split_digest"] == m2["split_digest"]


# --- report_by_category ----------------------------------------------------------------------------
def test_report_by_category_sums_and_split_breakdown_are_correct(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        r = _mk(tmp_path, f"real{i}.dxf", f"r{i}".encode(), "real_cls")
        r["data_origin"] = "real"   # declared real provenance (name alone would be "unknown")
        rows.append(r)
    for i in range(2):
        rows.append(_mk(tmp_path, f"part{i}_aug1.dxf", f"a{i}".encode(), "aug_cls"))
    rows.append(_mk(tmp_path, "widget_synth.dxf", b"synth-bytes", "synth_cls"))

    manifest = tm.build_versioned_manifest(
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.5
    )
    report = tm.report_by_category(manifest)

    assert report["total_rows"] == len(manifest["rows"]) == 6
    assert sum(report["by_category"].values()) == report["total_rows"]
    assert report["by_category"]["real"] == 3
    assert report["by_category"]["augmented"] == 2
    assert report["by_category"]["synthetic"] == 1

    # per-(category x split) and per-(category x class) both sum back to the per-category count
    for cat in tm.CATEGORIES:
        assert sum(report["by_category_split"].get(cat, {}).values()) == report["by_category"][cat]
        assert sum(report["by_category_taxonomy_v2_class"].get(cat, {}).values()) == report["by_category"][cat]

    # cross-check against the manifest rows directly: category x split matches row-level truth
    expected_split_by_cat = {c: {} for c in tm.CATEGORIES}
    expected_class_by_cat = {c: {} for c in tm.CATEGORIES}
    for row in manifest["rows"]:
        d = expected_split_by_cat[row["category"]]
        d[row["split"]] = d.get(row["split"], 0) + 1
        e = expected_class_by_cat[row["category"]]
        e[row["taxonomy_v2_class"]] = e.get(row["taxonomy_v2_class"], 0) + 1
    assert report["by_category_split"] == expected_split_by_cat
    assert report["by_category_taxonomy_v2_class"] == expected_class_by_cat


# --- verify_manifest: tamper / drift -> RED --------------------------------------------------------
def test_verify_manifest_passes_when_unchanged(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    tm.verify_manifest(rows, manifest)  # no raise


def test_verify_manifest_red_when_row_content_changes(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    # mutate bytes of one file AFTER the manifest was built -> content_hash drifts -> RED
    Path(rows[0]["file_path"]).write_bytes(b"TAMPERED-CONTENT")
    with pytest.raises(tm.IntegrityError, match="reproducibility check FAILED"):
        tm.verify_manifest(rows, manifest)


def test_verify_manifest_red_when_a_row_is_added(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    rows2 = list(rows) + [_mk(tmp_path, "brand_new_family.dxf", b"new-bytes", "cls")]
    with pytest.raises(tm.IntegrityError, match="reproducibility check FAILED"):
        tm.verify_manifest(rows2, manifest)


def test_verify_manifest_red_when_stored_digest_tampered(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    manifest["manifest_digest"] = "deadbeef" * 8
    with pytest.raises(tm.IntegrityError, match="reproducibility check FAILED"):
        tm.verify_manifest(rows, manifest)


# --- CLI round trip (build -> report -> verify) -----------------------------------------------------
def test_cli_build_report_verify_round_trip(tmp_path: Path) -> None:
    a = tmp_path / "a.dxf"
    a.write_bytes(b"AAA")
    b = tmp_path / "b_aug1.dxf"
    b.write_bytes(b"BBB")
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "cache_path", "taxonomy_v2_class"])
        writer.writeheader()
        writer.writerow({"file_path": str(a), "cache_path": "", "taxonomy_v2_class": "gearA"})
        writer.writerow({"file_path": str(b), "cache_path": "", "taxonomy_v2_class": "gearA"})

    out_path = tmp_path / "out.json"
    rc = tm.main(
        [
            "build",
            "--manifest",
            str(csv_path),
            "--source",
            "acme",
            "--license",
            "MIT",
            "--label-authority",
            "manifest:taxonomy_v2_class",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "evaluation-integrity-manifest-v2"
    assert len(manifest["rows"]) == 2

    rc = tm.main(["report", "--manifest-json", str(out_path)])
    assert rc == 0

    rc = tm.main(["verify", "--manifest", str(csv_path), "--manifest-json", str(out_path)])
    assert rc == 0

    # tamper the on-disk manifest json digest -> CLI verify must exit non-zero
    manifest["manifest_digest"] = "deadbeef" * 8
    out_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc = tm.main(["verify", "--manifest", str(csv_path), "--manifest-json", str(out_path)])
    assert rc == 1
