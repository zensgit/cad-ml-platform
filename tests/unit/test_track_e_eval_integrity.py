"""Track E slice-1 — leakage-safe splitter + reproducibility digest + gate-conformant artifact.

Torch-free: exercises the integrity core (content-hash + normalized-family split, conflict
quarantine, fail-closed unreadable content, deterministic digest, and the §8.1 exit-condition
reproducibility check whose `verify` goes RED on split tamper / duplicate reintroduction).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import track_e_eval_integrity as te
from scripts.eval_integrity_gate import REQUIRED_METRIC_KEYS, validate_artifact


def _mk(tmp_path: Path, name: str, content: bytes, label: str) -> dict:
    p = tmp_path / name
    p.write_bytes(content)
    return {"file_path": str(p), "cache_path": "", "taxonomy_v2_class": label}


def _metrics() -> dict:
    return {k: ({"gear": 0.9} if k == "per_class" else 0.03) for k in REQUIRED_METRIC_KEYS}


# --- normalized-family -------------------------------------------------------------------------
@pytest.mark.parametrize("name,fam", [
    ("gear.dxf", "gear"),
    ("gear_aug1.dxf", "gear"),
    ("gear_rot90.dxf", "gear"),
    ("gear_v2.dxf", "gear"),
    ("gear_aug3_rot270.dxf", "gear"),
    ("bracket-2.dxf", "bracket"),
    # adversarial-review Finding 1: under-collapse patterns must now collapse
    ("gear2.dxf", "gear"),            # bare trailing digit (no separator)
    ("gear (1).dxf", "gear"),         # macOS/Windows OS duplicate
    ("gear(3).dxf", "gear"),
    ("gear - Copy.dxf", "gear"),      # Windows " - Copy"
])
def test_normalized_family_collapses_variants(name: str, fam: str) -> None:
    assert te.normalized_family(name) == fam


def test_unicode_nfc_nfd_family_is_stable() -> None:
    import unicodedata
    nfc = unicodedata.normalize("NFC", "café.dxf")
    nfd = unicodedata.normalize("NFD", "café.dxf")
    assert te.normalized_family(nfc) == te.normalized_family(nfd)


def test_bare_digit_variants_share_a_component(tmp_path: Path) -> None:
    # Finding 1 end-to-end: gear.dxf / gear2.dxf / gear (1).dxf (DIFFERENT bytes, so content-hash
    # cannot rescue) must land on ONE side, at the default holdout fraction.
    rows = [
        _mk(tmp_path, "gear.dxf", b"A", "gear"),
        _mk(tmp_path, "gear2.dxf", b"B", "gear"),
        _mk(tmp_path, "gear (1).dxf", b"C", "gear"),
    ]
    for fam in ("p", "q", "r", "s", "t"):  # padding families
        rows.append(_mk(tmp_path, f"{fam}.dxf", fam.encode(), fam))
    split = te.compute_split(rows, holdout_fraction=te.DEFAULT_HOLDOUT_FRACTION)
    sides = {split["assignment"][r["file_path"]] for r in rows[:3]}
    assert len(sides) == 1, f"gear variants straddled the split: {sides}"


def test_declared_family_column_is_authoritative(tmp_path: Path) -> None:
    # differently-named files with the SAME declared family must never straddle, even when the
    # filename heuristic would not collapse them.
    a = _mk(tmp_path, "weird_alpha.dxf", b"1", "gear"); a["family"] = "GEAR-7"
    b = _mk(tmp_path, "totally_other.dxf", b"2", "gear"); b["family"] = "gear-7"
    rows = [a, b] + [_mk(tmp_path, f"{f}.dxf", f.encode(), f) for f in ("m", "n", "o", "p")]
    split = te.compute_split(rows, holdout_fraction=0.5)
    assert split["assignment"][a["file_path"]] == split["assignment"][b["file_path"]]


# --- fail-closed content hashing ---------------------------------------------------------------
def test_unreadable_content_is_quarantined_not_distinct(tmp_path: Path) -> None:
    good = _mk(tmp_path, "a.dxf", b"AAAA", "gear")
    missing = {"file_path": str(tmp_path / "does_not_exist.dxf"), "cache_path": "", "taxonomy_v2_class": "gear"}
    split = te.compute_split([good, missing])
    # the unreadable row must NOT appear as a split assignment (it is quarantined, not "distinct")
    assert missing["file_path"] not in split["assignment"]
    assert any("unreadable" in q["reason"] for q in split["quarantined"])


def test_nul_byte_path_is_quarantined_not_crash(tmp_path: Path) -> None:
    # adversarial-review Finding 2: an embedded NUL raises ValueError (not OSError); it must be
    # caught and quarantined, never crash compute_split.
    good = _mk(tmp_path, "a.dxf", b"AAAA", "gear")
    nul = {"file_path": "bad\x00path.dxf", "cache_path": "", "taxonomy_v2_class": "gear"}
    split = te.compute_split([good, nul])  # must not raise
    assert "bad\x00path.dxf" not in split["assignment"]
    assert any("bad" in q["file_path"] for q in split["quarantined"])


# --- no family / component straddles the split -------------------------------------------------
def test_no_family_straddles_train_and_holdout(tmp_path: Path) -> None:
    rows = []
    for fam in ("gear", "bracket", "shaft", "flange", "washer", "bolt"):
        rows.append(_mk(tmp_path, f"{fam}.dxf", f"{fam}-base".encode(), fam))
        rows.append(_mk(tmp_path, f"{fam}_aug1.dxf", f"{fam}-aug".encode(), fam))
    split = te.compute_split(rows, holdout_fraction=0.5)
    by_family: dict[str, set] = {}
    for row in rows:
        fam = te.normalized_family(row["file_path"])
        by_family.setdefault(fam, set()).add(split["assignment"][row["file_path"]])
    for fam, sides in by_family.items():
        assert len(sides) == 1, f"family {fam} straddled the split: {sides}"


def test_identical_content_across_families_lands_on_one_side(tmp_path: Path) -> None:
    # two DIFFERENT family names but byte-identical content -> unioned -> same side (no leak)
    dup = b"IDENTICAL-BYTES"
    rows = [_mk(tmp_path, "x.dxf", dup, "gear"), _mk(tmp_path, "y.dxf", dup, "gear")]
    # pad with other families so the split is non-trivial
    for fam in ("a", "b", "c", "d"):
        rows.append(_mk(tmp_path, f"{fam}.dxf", fam.encode(), fam))
    split = te.compute_split(rows, holdout_fraction=0.5)
    assert split["assignment"][rows[0]["file_path"]] == split["assignment"][rows[1]["file_path"]]


# --- conflict quarantine -----------------------------------------------------------------------
def test_identical_content_inconsistent_labels_is_quarantined(tmp_path: Path) -> None:
    dup = b"SAME-CONTENT"
    a = _mk(tmp_path, "p.dxf", dup, "gear")
    b = _mk(tmp_path, "q.dxf", dup, "bracket")   # same content, different label
    split = te.compute_split([a, b])
    assert a["file_path"] not in split["assignment"]
    assert b["file_path"] not in split["assignment"]
    assert any("label-conflict" in q["reason"] for q in split["quarantined"])


# --- determinism -------------------------------------------------------------------------------
def test_split_is_deterministic(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"cls{i%3}") for i in range(12)]
    d1 = te.split_digest(te.compute_split(rows))
    d2 = te.split_digest(te.compute_split(list(reversed(rows))))  # order-independent
    assert d1 == d2


# --- gate-conformant artifact ------------------------------------------------------------------
def test_build_artifact_is_gate_conformant(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), f"cls{i%2}") for i in range(6)]
    art = te.build_artifact(rows, _metrics())
    # the producer already asserts this internally; assert again against the real gate
    p = tmp_path / "art.json"
    p.write_text(__import__("json").dumps(art), encoding="utf-8")
    validate_artifact(str(p))  # raises GateError if not conformant
    assert art["schema_version"] == "evaluation-integrity-v2"
    assert art["split_strategy"] == "content-hash+normalized-family"


def test_build_artifact_rejects_metrics_missing_a_family(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, "f.dxf", b"c", "cls")]
    bad = _metrics()
    del bad["macro_f1"]
    with pytest.raises(te.IntegrityError):
        te.build_artifact(rows, bad)


# --- reproducibility exit-condition: tamper -> RED (the discrimination proof) -------------------
def test_verify_passes_when_unchanged(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(6)]
    art = te.build_artifact(rows, _metrics())
    te.verify_reproducible(rows, art)  # no raise


def test_verify_red_when_split_digest_tampered(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(6)]
    art = te.build_artifact(rows, _metrics())
    art["split_digest"] = "deadbeef" * 8   # tamper
    with pytest.raises(te.IntegrityError, match="reproducibility check FAILED"):
        te.verify_reproducible(rows, art)


def test_verify_red_when_duplicate_content_reintroduced(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(6)]
    art = te.build_artifact(rows, _metrics())
    # reintroduce duplicate content in a new family -> unions a component -> split changes -> RED
    rows2 = list(rows) + [_mk(tmp_path, "dup.dxf", b"c0", "cls")]  # same bytes as f0.dxf
    with pytest.raises(te.IntegrityError, match="reproducibility check FAILED"):
        te.verify_reproducible(rows2, art)


def test_verify_red_when_a_row_moves_family(tmp_path: Path) -> None:
    rows = [_mk(tmp_path, f"f{i}.dxf", f"c{i}".encode(), "cls") for i in range(6)]
    art = te.build_artifact(rows, _metrics())
    rows2 = list(rows) + [_mk(tmp_path, "brand_new_family.dxf", b"new", "cls")]
    with pytest.raises(te.IntegrityError, match="reproducibility check FAILED"):
        te.verify_reproducible(rows2, art)
