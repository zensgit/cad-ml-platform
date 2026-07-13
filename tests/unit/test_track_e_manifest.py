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
        root=tmp_path,
    )

    assert manifest["schema_version"] == "evaluation-integrity-manifest-v2"
    assert manifest["source"] == "acme-archive-2026"
    assert manifest["license"] == "proprietary-internal"
    assert manifest["label_authority"] == "manifest:taxonomy_v2_class"
    assert manifest["holdout_fraction"] == tm.DEFAULT_HOLDOUT_FRACTION

    # the unreadable row is EXCLUDED from rows and surfaced in quarantined instead
    assert len(manifest["rows"]) == 4
    locators = {r["locator"] for r in manifest["rows"]}
    assert "does_not_exist.dxf" not in locators
    # absolute run paths never enter the manifest
    for r in manifest["rows"]:
        assert not r["locator"].startswith("/")
    assert any(q["reason_code"] == "unreadable" for q in manifest["quarantined"])

    required_fields = {
        "locator",
        "cache_locator",
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

    by_loc = {r["locator"]: r for r in manifest["rows"]}
    assert by_loc["gear.dxf"]["category"] == "real"                      # declared data_origin
    assert by_loc["gear_aug1.dxf"]["category"] == "augmented"
    assert by_loc["part_synth.dxf"]["category"] == "synthetic"
    assert by_loc["bracket.dxf"]["category"] == "unknown"                # unmarked -> unknown

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
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.5,
        root=tmp_path,
    )
    by_loc = {r["locator"]: r for r in manifest["rows"]}
    # same declared family -> same family key -> can never straddle the split (inherited from slice-1)
    assert by_loc["weird_alpha.dxf"]["family"] == by_loc["totally_other.dxf"]["family"]
    assert by_loc["weird_alpha.dxf"]["split"] == by_loc["totally_other.dxf"]["split"]


# --- manifest_digest determinism ------------------------------------------------------------------
def test_manifest_digest_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"cls{i % 3}") for i in range(8)]
    m1 = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    m2 = tm.build_versioned_manifest(list(reversed(rows)), source="s", license_="l", label_authority="a", root=tmp_path)
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
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.5,
        root=tmp_path,
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
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    tm.verify_manifest(rows, manifest, root=tmp_path)  # no raise


def test_verify_manifest_red_when_row_content_changes(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    # mutate bytes of one file AFTER the manifest was built -> content_hash drifts -> RED
    Path(rows[0]["file_path"]).write_bytes(b"TAMPERED-CONTENT")
    with pytest.raises(tm.IntegrityError, match="FAILED"):
        tm.verify_manifest(rows, manifest, root=tmp_path)


def test_verify_manifest_red_when_a_row_is_added(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    rows2 = list(rows) + [_mk(tmp_path, "brand_new_family.dxf", b"new-bytes", "cls")]
    with pytest.raises(tm.IntegrityError, match="FAILED"):
        tm.verify_manifest(rows2, manifest, root=tmp_path)


def test_verify_manifest_red_when_stored_digest_tampered(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(5)]
    manifest = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    manifest["manifest_digest"] = "deadbeef" * 8
    with pytest.raises(tm.IntegrityError, match="FAILED"):
        tm.verify_manifest(rows, manifest, root=tmp_path)


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
            "--root",
            str(tmp_path),
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

    rc = tm.main(["verify", "--manifest", str(csv_path), "--root", str(tmp_path), "--manifest-json", str(out_path)])
    assert rc == 0

    # tamper the on-disk manifest json digest -> CLI verify must exit non-zero
    manifest["manifest_digest"] = "deadbeef" * 8
    out_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc = tm.main(["verify", "--manifest", str(csv_path), "--root", str(tmp_path), "--manifest-json", str(out_path)])
    assert rc == 1


def test_report_missing_or_illegal_category_maps_to_unknown_not_real() -> None:
    # review: a missing or illegal category must NOT default to "real". It reports as "unknown",
    # and illegal values are surfaced separately.
    manifest = {"rows": [
        {"split": "train", "taxonomy_v2_class": "c"},                    # missing category
        {"split": "train", "taxonomy_v2_class": "c", "category": "bogus"},  # illegal
        {"split": "holdout", "taxonomy_v2_class": "c", "category": "real"},
    ]}
    rep = tm.report_by_category(manifest)
    assert rep["by_category"]["real"] == 1
    assert rep["by_category"]["unknown"] == 2      # missing + illegal both -> unknown
    assert rep["illegal_category_rows"] == 1
    assert sum(rep["by_category"].values()) == rep["total_rows"] == 3


@pytest.mark.parametrize("field,newval", [
    ("schema_version", "hacked-v9"),
    ("provenance_complete", True),
    ("unknown_provenance_rows", 0),
    ("quarantined", []),                       # a manifest WITH a quarantined row -> [] is a tamper
    ("source", "evil-source"),
])
def test_verify_manifest_red_on_envelope_tamper(tmp_path: Path, field, newval) -> None:
    # review P1: tampering ANY envelope field (not just rows) must be caught by the digest.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(4)]  # unmarked -> unknown
    # include an unreadable row so `quarantined` is non-empty (makes the quarantined-tamper real)
    rows.append({"file_path": str(tmp_path / "missing.dxf"), "cache_path": "", "taxonomy_v2_class": "cls"})
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    assert man["quarantined"], "test setup: expected a quarantined row"
    man[field] = newval                       # tamper, leave manifest_digest as-is
    with pytest.raises(tm.IntegrityError):
        tm.verify_manifest(rows, man, root=tmp_path)


@pytest.mark.parametrize("bad", [
    {"source": "", "license_": "l", "label_authority": "a"},
    {"source": "s", "license_": "   ", "label_authority": "a"},
    {"source": "s", "license_": "l", "label_authority": ""},
])
def test_empty_provenance_fails_closed(tmp_path: Path, bad) -> None:
    rows = [_mk(tmp_path, "f.dxf", b"c", "cls")]
    with pytest.raises(tm.IntegrityError):
        tm.build_versioned_manifest(rows, **bad, root=tmp_path)


def test_manifest_digest_is_fresh_clone_stable(tmp_path: Path) -> None:
    # identical bytes + basenames under two DIFFERENT absolute roots -> same manifest_digest
    # (keyed on sample_id, not the host path).
    def _clone(root: Path) -> str:
        root.mkdir()
        rows = []
        for name, b, lbl in [("gear.dxf", b"A", "g"), ("bolt.dxf", b"B", "b")]:
            (root / name).write_bytes(b)
            rows.append({"file_path": str(root / name), "cache_path": "", "taxonomy_v2_class": lbl})
        return tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=root)["manifest_digest"]
    assert _clone(tmp_path / "cloneA") == _clone(tmp_path / "cloneB")


def test_verify_red_on_stored_locator_tamper(tmp_path: Path) -> None:
    # locators are digested: a naive stored-locator redirect trips the ENVELOPE check.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(4)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["rows"][0]["locator"] = "evil_redirect.dxf"
    with pytest.raises(tm.IntegrityError, match="FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_verify_red_on_redigested_locator_tamper(tmp_path: Path) -> None:
    # a SOPHISTICATED tamper that also recomputes manifest_digest passes the envelope check but
    # must be caught by the (sample_id, locator) binding against the re-derived rows.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(4)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["rows"][0]["locator"] = "evil_redirect.dxf"
    man["manifest_digest"] = tm._manifest_digest(man)   # attacker re-digests the envelope
    with pytest.raises(tm.IntegrityError, match="row binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_verify_red_on_redigested_cache_locator_tamper(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(4)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["rows"][0]["cache_locator"] = "evil/cache.pt"
    man["manifest_digest"] = tm._manifest_digest(man)
    with pytest.raises(tm.IntegrityError, match="row binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_manifest_digest_stable_across_clones_with_quarantined_rows(tmp_path: Path) -> None:
    # review P1: quarantine records must not leak absolute paths / OS text into the digest — the
    # SAME missing file under two different clone roots must yield the same manifest_digest.
    def _clone(root: Path) -> str:
        root.mkdir()
        (root / "good.dxf").write_bytes(b"G")
        rows = [
            {"file_path": str(root / "good.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
            {"file_path": str(root / "gone.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
        ]
        m = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=root)
        assert m["quarantined"] and m["quarantined"][0]["reason_code"] == "unreadable"
        assert m["quarantined"][0]["locator"] == "gone.dxf"
        return m["manifest_digest"]
    assert _clone(tmp_path / "cloneA") == _clone(tmp_path / "cloneB")


def _clone_rows(root: Path) -> list:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sub").mkdir(exist_ok=True)
    rows = []
    for rel, b, lbl in [("gear.dxf", b"A", "g"), ("sub/gear.dxf", b"B", "s"), ("bolt.dxf", b"C", "b")]:
        p = root / rel
        p.write_bytes(b)
        rows.append({"file_path": str(p), "cache_path": "", "taxonomy_v2_class": lbl})
    return rows


def test_artifact_built_on_clone_a_verifies_on_clone_b(tmp_path: Path) -> None:
    # THE fresh-clone portability discriminator (review P1): build on clone A, verify the SAME
    # artifact against clone B's rows (same bytes + layout, different absolute root) -> PASS.
    man = tm.build_versioned_manifest(
        _clone_rows(tmp_path / "cloneA"), source="s", license_="l", label_authority="a",
        root=tmp_path / "cloneA",
    )
    tm.verify_manifest(_clone_rows(tmp_path / "cloneB"), man, root=tmp_path / "cloneB")  # must not raise


def test_same_basename_different_dirs_do_not_collide(tmp_path: Path) -> None:
    # locators are full relative paths, not basenames: gear.dxf vs sub/gear.dxf stay distinct.
    man = tm.build_versioned_manifest(
        _clone_rows(tmp_path / "cloneC"), source="s", license_="l", label_authority="a",
        root=tmp_path / "cloneC",
    )
    locs = sorted(r["locator"] for r in man["rows"])
    assert locs == ["bolt.dxf", "gear.dxf", "sub/gear.dxf"]


def test_quarantine_locator_is_relative_full_path(tmp_path: Path) -> None:
    root = tmp_path / "cloneQ"
    rows = _clone_rows(root)
    rows.append({"file_path": str(root / "sub" / "gone.dxf"), "cache_path": "", "taxonomy_v2_class": "g"})
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=root)
    q = man["quarantined"][0]
    assert q["locator"] == "sub/gone.dxf"      # relative full path, not a basename
    assert "file_path" not in q                # no absolute run path in the manifest


# --- containment: root escape / absolute / .. locators are rejected (review) --------------------
def test_file_outside_explicit_root_fails_closed(tmp_path: Path) -> None:
    inside = tmp_path / "data"; inside.mkdir()
    (inside / "a.dxf").write_bytes(b"A")
    outside = tmp_path / "elsewhere.dxf"; outside.write_bytes(b"B")
    rows = [
        {"file_path": str(inside / "a.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
        {"file_path": str(outside), "cache_path": "", "taxonomy_v2_class": "g"},
    ]
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=inside)


def test_cache_path_outside_root_fails_closed(tmp_path: Path) -> None:
    inside = tmp_path / "data"; inside.mkdir()
    (inside / "a.dxf").write_bytes(b"A")
    rows = [{"file_path": str(inside / "a.dxf"), "cache_path": "../evil_cache.pt", "taxonomy_v2_class": "g"}]
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=inside)


def test_relative_dotdot_input_row_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "a.dxf").write_bytes(b"A")
    rows = [
        {"file_path": str(tmp_path / "a.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
        {"file_path": "../sneaky.dxf", "cache_path": "", "taxonomy_v2_class": "g"},
    ]
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root|unreadable"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)


@pytest.mark.parametrize("evil", ["../outside.dxf", "/abs/path.dxf", "a/../../b.dxf"])
def test_stored_hostile_locator_rejected_even_if_redigested(tmp_path: Path, evil: str) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(3)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["rows"][0]["locator"] = evil
    man["manifest_digest"] = tm._manifest_digest(man)   # attacker re-digests
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root|absolute"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_absolute_rows_without_explicit_root_fail_closed(tmp_path: Path) -> None:
    # THE review bypass repro: /X/dataset/a.dxf + /X/outside/secret.dxf with NO root previously let
    # a common-parent heuristic widen the root to /X and legitimize BOTH ("dataset/a.dxf" +
    # "outside/secret.dxf"). Root inference is now forbidden: ANY absolute input without an
    # explicit root is RED — the trust boundary is declared, never derived from the data.
    dataset = tmp_path / "dataset"; dataset.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (dataset / "a.dxf").write_bytes(b"A")
    (outside / "secret.dxf").write_bytes(b"S")
    rows = [
        {"file_path": str(dataset / "a.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
        {"file_path": str(outside / "secret.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
    ]
    with pytest.raises(tm.IntegrityError, match="explicit root is required"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")
    # and with the CORRECT explicit root, the stray outside file is rejected as an escape — never
    # widened in by inferring a broader root.
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(
            rows, source="s", license_="l", label_authority="a", root=dataset
        )


def test_no_root_single_dir_absolute_also_requires_explicit_root(tmp_path: Path) -> None:
    # even a clean single-dir absolute input must not silently self-infer a root.
    (tmp_path / "a.dxf").write_bytes(b"A")
    rows = [{"file_path": str(tmp_path / "a.dxf"), "cache_path": "", "taxonomy_v2_class": "g"}]
    with pytest.raises(tm.IntegrityError, match="explicit"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a")


def test_symlink_escape_is_contained(tmp_path: Path) -> None:
    # resolve()-based containment: a symlink inside the root pointing OUTSIDE must be rejected.
    import os
    dataset = tmp_path / "dataset"; dataset.mkdir()
    (dataset / "real.dxf").write_bytes(b"R")
    secret = tmp_path / "secret.dxf"; secret.write_bytes(b"S")
    link = dataset / "link.dxf"
    try:
        os.symlink(str(secret), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rows = [
        {"file_path": str(dataset / "real.dxf"), "cache_path": "", "taxonomy_v2_class": "g"},
        {"file_path": str(link), "cache_path": "", "taxonomy_v2_class": "g"},
    ]
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=dataset)


def test_verify_red_on_redigested_provenance_flip(tmp_path: Path) -> None:
    # A re-digesting attacker flips provenance_complete False->True (hiding unknown-provenance rows)
    # and recomputes manifest_digest. The envelope self-check passes, but provenance binding — which
    # re-derives provenance_complete/unknown_provenance_rows FROM the rows — must catch it.
    real = _mk(tmp_path, "gear.dxf", b"A", "g"); real["data_origin"] = "real"
    unknown = _mk(tmp_path, "mystery.dxf", b"B", "m")   # unmarked -> unknown provenance
    rows = [real, unknown]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    assert man["provenance_complete"] is False and man["unknown_provenance_rows"] == 1
    man["provenance_complete"] = True                    # attacker forges a clean-provenance verdict
    man["unknown_provenance_rows"] = 0
    man["manifest_digest"] = tm._manifest_digest(man)    # ...and re-digests to pass self-consistency
    with pytest.raises(tm.IntegrityError, match="provenance binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_relative_symlink_escape_is_contained(tmp_path: Path) -> None:
    # review P1: the relative-path branch previously skipped resolve() containment. A RELATIVE
    # locator that is a symlink to OUTSIDE root (read as root/link.dxf by content_hash) must be RED.
    import os
    dataset = tmp_path / "dataset"; dataset.mkdir()
    (dataset / "real.dxf").write_bytes(b"R")
    secret = tmp_path / "secret.dxf"; secret.write_bytes(b"S")
    try:
        os.symlink(str(secret), str(dataset / "link.dxf"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rows = [
        {"file_path": "real.dxf", "cache_path": "", "taxonomy_v2_class": "g"},   # relative, in-root
        {"file_path": "link.dxf", "cache_path": "", "taxonomy_v2_class": "g"},   # relative symlink OUT
    ]
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=dataset)


@pytest.mark.parametrize("field,forged", [
    ("category", "real"),          # unknown -> real (forges clean provenance)
    ("taxonomy_v2_class", "forged"),
    ("locator", "evil_redirect.dxf"),
    ("content_hash", "0" * 64),
])
def test_verify_red_on_redigested_per_row_tamper(tmp_path: Path, field: str, forged: str) -> None:
    # review P1: provenance binding only guarded the top-level verdict; a re-digested PER-ROW tamper
    # to category/split/taxonomy/locator/content_hash slipped through. Full row binding must catch it.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "g") for i in range(4)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["rows"][0][field] = forged
    man["manifest_digest"] = tm._manifest_digest(man)     # attacker re-digests
    with pytest.raises(tm.IntegrityError, match="row binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_verify_red_on_redigested_quarantine_tamper(tmp_path: Path) -> None:
    # a re-digested drop/re-label of a quarantined row must be RED.
    good = _mk(tmp_path, "good.dxf", b"G", "g")
    rows = [good, {"file_path": str(tmp_path / "gone.dxf"), "cache_path": "", "taxonomy_v2_class": "g"}]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    assert man["quarantined"], "expected a quarantined missing row"
    man["quarantined"][0]["reason_code"] = "other"        # re-label the quarantine reason
    man["manifest_digest"] = tm._manifest_digest(man)
    with pytest.raises(tm.IntegrityError, match="quarantine binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_verify_red_on_redigested_split_flip(tmp_path: Path) -> None:
    # moving a sample across the split (train<->holdout) + re-digest must be RED via row binding.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "g") for i in range(6)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    cur = man["rows"][0]["split"]
    man["rows"][0]["split"] = "train" if cur == "holdout" else "holdout"   # flip to the opposite
    man["manifest_digest"] = tm._manifest_digest(man)
    with pytest.raises(tm.IntegrityError, match="row binding FAILED"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_escaping_symlink_rejected_before_any_content_read(tmp_path: Path, monkeypatch) -> None:
    # review P1: containment must fire BEFORE compute_split/content_hash opens the out-of-root file.
    import os
    dataset = tmp_path / "dataset"; dataset.mkdir()
    (dataset / "real.dxf").write_bytes(b"R")
    secret = tmp_path / "secret.dxf"; secret.write_bytes(b"S")
    try:
        os.symlink(str(secret), str(dataset / "link.dxf"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    rows = [
        {"file_path": "real.dxf", "cache_path": "", "taxonomy_v2_class": "g"},
        {"file_path": "link.dxf", "cache_path": "", "taxonomy_v2_class": "g"},   # symlink OUT of root
    ]
    calls = {"split": 0}
    real_split = tm.compute_split
    monkeypatch.setattr(
        tm, "compute_split",
        lambda *a, **k: (calls.__setitem__("split", calls["split"] + 1) or real_split(*a, **k)),
    )
    with pytest.raises(tm.IntegrityError, match="escapes the dataset root"):
        tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=dataset)
    assert calls["split"] == 0   # containment pre-flight fired BEFORE compute_split/content_hash


def test_verify_red_on_redigested_schema_version(tmp_path: Path) -> None:
    # review P1: verify must PIN schema_version, not trust the artifact's self-declared value.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "g") for i in range(4)]
    man = tm.build_versioned_manifest(rows, source="s", license_="l", label_authority="a", root=tmp_path)
    man["schema_version"] = "attacker-schema-v999"
    man["manifest_digest"] = tm._manifest_digest(man)   # re-digest to pass self-consistency
    with pytest.raises(tm.IntegrityError, match="schema_version mismatch"):
        tm.verify_manifest(rows, man, root=tmp_path)


def test_verify_red_on_redigested_holdout_policy(tmp_path: Path) -> None:
    # review P1: an artifact rebuilt under a different split policy (0.9) + full re-digest must be RED
    # because verify measures against the TRUSTED policy (default 0.2), not the artifact's claim.
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"c{i}") for i in range(20)]
    man = tm.build_versioned_manifest(
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.9, root=tmp_path
    )  # fully self-consistent artifact, just a hostile split policy
    with pytest.raises(tm.IntegrityError, match="holdout policy mismatch"):
        tm.verify_manifest(rows, man, root=tmp_path)   # trusted default 0.2 != declared 0.9


def test_verify_accepts_nondefault_policy_when_caller_declares_it(tmp_path: Path) -> None:
    # the trusted policy is caller-supplied: a legitimate non-default build verifies when the caller
    # passes the matching expected_holdout_fraction (proves the pin isn't just hardcoded to 0.2).
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"c{i}") for i in range(20)]
    man = tm.build_versioned_manifest(
        rows, source="s", license_="l", label_authority="a", holdout_fraction=0.3, root=tmp_path
    )
    tm.verify_manifest(rows, man, root=tmp_path, expected_holdout_fraction=0.3)  # must not raise
