#!/usr/bin/env python3
"""Track E slice-2: versioned evaluation-integrity manifest + real/synthetic/augmented reporting.

Builds on slice-1 (``scripts/track_e_eval_integrity.py``): PRODUCT_STRATEGY.md §8.1.5 requires
separate reporting for real, synthetic, and augmented data, and §8.1.6 requires a versioned
manifest carrying source, license, provenance, family, hash, split, and label authority. Slice-1
already delivers the leakage-safe split (content-hash + normalized-family, conflict quarantine,
fail-closed unreadable content, deterministic digest); this module REUSES those primitives rather
than re-deriving them, so a manifest built here can never drift from slice-1's split. (The L3 gate
is UNCONDITIONAL — it trusts no artifact and nothing here can unlock it; this is inspection/audit
tooling only.)

  * ``categorize`` — regex-based, case-insensitive, deterministic real/synthetic/augmented
    classification from filename/family markers (§8.1.5). No I/O, no RNG.
  * ``build_versioned_manifest`` — runs slice-1's ``compute_split`` (so quarantined rows are
    EXCLUDED and fail-closed behaviour is inherited unchanged) and emits one enriched record per
    surviving row carrying every §8.1.6 field.
  * ``report_by_category`` — counts per category, per (category × split), per
    (category × taxonomy_v2_class); always sums back to the row count.
  * ``verify_manifest`` — re-derives the manifest from the same rows and raises ``IntegrityError``
    (imported from slice-1) if ``manifest_digest`` or ``split_digest`` differs: tamper / drift
    detection, dry-run posture (non-blocking), same as slice-1's ``verify_reproducible``.

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Resolve the sibling slice-1 module whether this file is imported as
# ``scripts.track_e_manifest`` (pytest, repo root on sys.path) or run as
# ``python3 scripts/track_e_manifest.py`` (CLI, scripts/ on sys.path). Add the scripts/ dir so the
# top-level import works in both — same pattern slice-1 uses to reach ``eval_integrity_gate``.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reused, never re-derived — a manifest built here is split-conformant with slice-1 by
# construction (single source of truth for the leakage-safe split).
from track_e_eval_integrity import (  # noqa: E402  (sibling script; scripts/ ensured on sys.path above)
    IntegrityError,
    _family_key,
    compute_split,
    split_digest,
)

SCHEMA_VERSION = "evaluation-integrity-manifest-v2"
DEFAULT_HOLDOUT_FRACTION = 0.2

# §8.1.5: separate reporting for real, synthetic, and augmented data — plus "unknown", because the
# absence of a synthetic/augmentation marker does NOT prove a sample is real provenance (§8.1.5 is
# a provenance report, not an inference). Order is only the canonical report-scaffolding order.
CATEGORIES = ("real", "synthetic", "augmented", "unknown")

# Explicit provenance columns are AUTHORITATIVE — preferred over any filename inference.
_DECLARED_COLUMNS = ("data_origin", "provenance", "category")
_VALID_DECLARED = {"real", "synthetic", "augmented", "unknown"}

# Markers that denote a sample was produced by data augmentation of some other source sample.
# Anchored on a separator-or-start boundary before, and a separator-or-end boundary after any
# trailing digit run, so "rotator" / "generalpurpose" style substrings never false-positive.
_AUGMENTED_RE = re.compile(
    r"(?i)(?:^|[_\-\s])(?:augmented|augment|aug|rotated|rotate|rot|flipped|flip|"
    r"noised|noise|jittered|jitter|scaled|scale)\d*(?:[_\-\s]|$)"
)

# Markers that denote a sample was produced (not augmented) rather than captured from a real
# drawing: synthetic / generated / GAN output. Deliberately does NOT include a bare "gen" —
# spec §8.1.5 lists synth/synthetic/generated/gan; a bare "gen" abbreviation is not in that list
# and is common enough in unrelated filenames (e.g. "gen2_assembly") to risk false positives.
_SYNTHETIC_RE = re.compile(
    r"(?i)(?:^|[_\-\s])(?:synthetic|synth|generated|generate|gan)\d*(?:[_\-\s]|$)"
)


def categorize(row: dict) -> str:
    """Classify a manifest row's provenance: real / synthetic / augmented / unknown (§8.1.5).

    1. An explicit ``data_origin``/``provenance``/``category`` column is AUTHORITATIVE.
    2. Otherwise a positive filename/family marker can identify ``augmented`` or ``synthetic``
       (augmentation wins ties — an augmented copy is first an augmentation *of* something).
    3. Otherwise ``unknown`` — the absence of a marker is NOT proof of real provenance, so an
       unmarked, undeclared sample must not be silently counted as real (it keeps the manifest's
       provenance incomplete rather than fabricating a "real" label).

    Deterministic, no I/O, no RNG.
    """
    for col in _DECLARED_COLUMNS:
        val = str(row.get(col, "") or "").strip().lower()
        if val in _VALID_DECLARED:
            return val
    stem = Path(str(row.get("file_path", ""))).stem
    family = str(row.get("family", "") or "")
    text = f"{stem} {family}" if family else stem
    if _AUGMENTED_RE.search(text):
        return "augmented"
    if _SYNTHETIC_RE.search(text):
        return "synthetic"
    return "unknown"


def _dataset_root(rows: List[dict], root: Optional[Path]) -> Optional[Path]:
    """The base every locator is expressed against. Explicit ``root`` wins; otherwise the common
    parent of all ABSOLUTE row paths (deterministic given the same clone layout). ``None`` when all
    row paths are already relative (repo-style manifests) — they are then locators as-is."""
    if root is not None:
        return Path(root)
    import os as _os

    abs_dirs = [
        str(Path(str(r.get("file_path", ""))).parent)
        for r in rows
        if str(r.get("file_path", "")) and Path(str(r.get("file_path", ""))).is_absolute()
    ]
    return Path(_os.path.commonpath(abs_dirs)) if abs_dirs else None


def _relative_locator(fp: str, root: Optional[Path]) -> str:
    """Dataset-root-RELATIVE, NFC-normalized, POSIX-style locator.

    The sample's portable address inside the versioned dataset: it IS digested, must be identical
    on every fresh clone, and never carries a host prefix. A full relative path (not a basename),
    so same-named files in different directories cannot collide.
    """
    import os as _os
    import unicodedata as _ud

    p = Path(fp)
    if root is not None and p.is_absolute():
        rel = _os.path.relpath(str(p), str(root))
    else:
        rel = str(p)  # already relative -> use as-is
    return _ud.normalize("NFC", Path(rel).as_posix())


def _enrich_rows(
    rows: List[dict],
    split: dict,
    *,
    source: str,
    license_: str,
    label_authority: str,
    dataset_root: Optional[Path],
) -> List[dict]:
    """Build one §8.1.6-complete record per row that survived slice-1's ``compute_split``.

    Rows slice-1 quarantined (missing/unreadable content, label conflict, missing field) are
    silently absent from ``split["assignment"]`` and therefore excluded here too — fail-closed
    behaviour is inherited, not re-implemented. The ``content_hash`` is the SAME snapshot
    ``compute_split`` computed to build the split (``split["content_hashes"]``) — NOT a second read.

    ABSOLUTE RUN PATHS DO NOT ENTER THE MANIFEST (fresh-clone portability): each row carries a
    dataset-root-relative ``locator`` (+ ``cache_locator``), and locators ARE digested. A consumer
    resolves data as ``<its own dataset root>/<locator>``.
    """
    content_hashes = split["content_hashes"]
    enriched: List[dict] = []
    for row in rows:
        key = str(row.get("file_path", ""))
        if key not in split["assignment"]:
            continue  # quarantined upstream by compute_split; never enters the manifest
        ch = content_hashes[key]
        family = _family_key(row)
        label = str(row.get("taxonomy_v2_class", "")).strip()
        cache = str(row.get("cache_path", ""))
        enriched.append(
            {
                # host-INDEPENDENT stable identity (content + family + label)
                "sample_id": hashlib.sha256(f"{ch}|{family}|{label}".encode()).hexdigest()[:16],
                "locator": _relative_locator(key, dataset_root),
                "cache_locator": _relative_locator(cache, dataset_root) if cache else "",
                "taxonomy_v2_class": label,
                "family": family,
                "content_hash": ch,                    # the authoritative split-time snapshot
                "split": split["assignment"][key],
                "category": categorize(row),
                "source": source,
                "license": license_,
                "label_authority": label_authority,
            }
        )
    return enriched


# Free-text human detail is excluded from the digest (varies per clone/OS locale). Locators are
# dataset-root-relative and DO enter the digest; absolute run paths never enter the manifest.
_DIGEST_EXCLUDED_ROW_FIELDS = ("detail",)

# Stable quarantine reason codes (these DO enter the digest; the human `detail` does not).
_QUARANTINE_REASON_CODES = (
    ("unreadable content", "unreadable"),
    ("label-conflict", "label_conflict"),
    ("missing file_path or label", "missing_field"),
)


def _normalize_quarantine(entry: dict, dataset_root: Optional[Path]) -> dict:
    """Rewrite a slice-1 quarantine record into a fresh-clone-stable form.

    ``locator`` = dataset-root-RELATIVE path (a full relative path, not a basename, so same-named
    files in different directories cannot collide); ``reason_code`` = stable enum-like code. Both
    are digested. The OS error text is kept as human ``detail`` only — EXCLUDED from the digest —
    so two clones quarantining the same missing file produce the same ``manifest_digest``. No
    absolute run path enters the manifest.
    """
    fp = str(entry.get("file_path", ""))
    reason = str(entry.get("reason", ""))
    code = "other"
    for prefix, rc in _QUARANTINE_REASON_CODES:
        if reason.startswith(prefix):
            code = rc
            break
    return {
        "locator": _relative_locator(fp, dataset_root),
        "reason_code": code,
        "detail": reason,    # human detail — excluded from the digest
    }


# The digest covers the ENTIRE manifest envelope except the digest field itself, so tampering ANY
# field (schema_version, provenance_complete, unknown_provenance_rows, quarantined, rows, source,
# split_digest, …) is detected — not just a changed row. Rows are order-canonicalized first.
_DIGEST_EXCLUDED = ("manifest_digest",)


def _strip_host_fields(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _DIGEST_EXCLUDED_ROW_FIELDS}


def _canonicalize_for_digest(manifest: dict) -> dict:
    m = {k: v for k, v in manifest.items() if k not in _DIGEST_EXCLUDED}
    # rows/quarantined: drop host-bound path fields (fresh-clone stability via sample_id) and make
    # the list order-independent (row identity, not position, is what matters).
    if isinstance(m.get("rows"), list):
        m["rows"] = sorted(
            (_strip_host_fields(r) for r in m["rows"]),
            key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False),
        )
    if isinstance(m.get("quarantined"), list):
        m["quarantined"] = sorted(
            (_strip_host_fields(r) for r in m["quarantined"]),
            key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False),
        )
    return m


def _manifest_digest(manifest_without_digest: dict) -> str:
    """sha256 over the full canonicalized manifest envelope (minus ``manifest_digest``)."""
    payload = json.dumps(
        _canonicalize_for_digest(manifest_without_digest),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_versioned_manifest(
    rows: Iterable[dict],
    *,
    source: str,
    license_: str,
    label_authority: str,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    root: Optional[Path] = None,
) -> dict:
    """Build the §8.1.6 versioned manifest: source, license, provenance, family, hash, split,
    label authority — one enriched record per row that survives slice-1's leakage-safe split.

    Provenance metadata is required: ``source`` / ``license`` / ``label_authority`` must be non-empty
    after trimming (a manifest with blank provenance fails closed). Quarantined rows (§8.1.2
    conflicts, unreadable content, missing fields) are excluded from ``rows`` and surfaced separately.
    """
    source, license_, label_authority = source.strip(), license_.strip(), label_authority.strip()
    for name, val in (("source", source), ("license", license_), ("label_authority", label_authority)):
        if not val:
            raise IntegrityError(f"{name} must be a non-empty provenance value")

    rows = list(rows)
    split = compute_split(rows, holdout_fraction=holdout_fraction, root=root)
    dataset_root = _dataset_root(rows, root)
    enriched = _enrich_rows(
        rows, split, source=source, license_=license_, label_authority=label_authority,
        dataset_root=dataset_root,
    )
    unknown_provenance = sum(1 for r in enriched if r["category"] == "unknown")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "license": license_,
        "label_authority": label_authority,
        "holdout_fraction": holdout_fraction,
        # §8.1.5 provenance completeness: any row of unknown provenance means the manifest is NOT a
        # clean, fully-attributed dataset — a downstream evaluation must NOT treat it as complete.
        "provenance_complete": unknown_provenance == 0,
        "unknown_provenance_rows": unknown_provenance,
        "rows": enriched,
        "quarantined": [_normalize_quarantine(q, dataset_root) for q in split["quarantined"]],
        "split_digest": split_digest(split),
    }
    manifest["manifest_digest"] = _manifest_digest(manifest)  # over the full envelope above
    return manifest


def _report_category(row: dict) -> str:
    """The category to report a row under. A missing or ILLEGAL category is NOT defaulted to
    "real" — an unattributed row must never inflate the trusted-provenance count (review). It maps
    to "unknown", the honest bucket for "we don't know this is real"."""
    cat = str(row.get("category", "") or "")
    return cat if cat in CATEGORIES else "unknown"


def report_by_category(manifest: dict) -> dict:
    """Per-category (§8.1.5) breakdown: counts per category, per (category × split), and per
    (category × taxonomy_v2_class). Every sub-breakdown sums back to the total row count. A row with
    a missing/illegal category is reported as "unknown", never silently "real"."""
    rows = manifest.get("rows", [])
    by_category: Dict[str, int] = {c: 0 for c in CATEGORIES}
    by_category_split: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}
    by_category_class: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}
    illegal = 0

    for row in rows:
        raw = str(row.get("category", "") or "")
        if raw and raw not in CATEGORIES:
            illegal += 1
        cat = _report_category(row)
        by_category[cat] += 1

        split = row.get("split", "")
        cat_split = by_category_split[cat]
        cat_split[split] = cat_split.get(split, 0) + 1

        cls = row.get("taxonomy_v2_class", "")
        cat_class = by_category_class[cat]
        cat_class[cls] = cat_class.get(cls, 0) + 1

    return {
        "total_rows": len(rows),
        "illegal_category_rows": illegal,   # surfaced, not hidden — a data-integrity signal
        "by_category": by_category,
        "by_category_split": by_category_split,
        "by_category_taxonomy_v2_class": by_category_class,
    }


def verify_manifest(rows: Iterable[dict], manifest: dict, *, root: Optional[Path] = None) -> None:
    """Confirm the manifest is untampered and its split still re-derives. RAISES ``IntegrityError``.

    Two independent checks:
      1. **Envelope self-consistency** — recompute ``manifest_digest`` over the STORED envelope
         (minus the digest itself) and compare to the stored value. This catches a tamper to ANY
         field — schema_version, provenance_complete, unknown_provenance_rows, quarantined, rows,
         provenance metadata — not just a changed row.
      2. **Split drift** — re-derive the split from the actual ``rows`` and compare ``split_digest``,
         so a change in the underlying file content (or an added/removed row) is caught.
      3. **Locator binding** — the stored portable ``(sample_id, locator)`` pairs must equal the
         pairs re-derived from the actual rows, so a redirected locator is RED.

    FRESH-CLONE PORTABLE: locators are dataset-root-relative, so an artifact built on clone A
    verifies against clone B's rows (same bytes, same layout, different absolute root).
    Dry-run posture (non-blocking): the caller decides what to do with the raised error.
    """
    stored_manifest_digest = manifest.get("manifest_digest")
    envelope_digest = _manifest_digest(manifest)  # over the stored envelope, minus manifest_digest
    if stored_manifest_digest != envelope_digest:
        raise IntegrityError(
            "manifest envelope FAILED self-consistency: manifest_digest "
            f"{str(stored_manifest_digest)[:12]} != content digest {envelope_digest[:12]} — a field "
            "(schema_version / provenance_complete / quarantined / rows / …) was tampered."
        )

    recomputed = build_versioned_manifest(
        rows,
        source=str(manifest.get("source", "")),
        license_=str(manifest.get("license", "")),
        label_authority=str(manifest.get("label_authority", "")),
        holdout_fraction=manifest.get("holdout_fraction", DEFAULT_HOLDOUT_FRACTION),
        root=root,
    )
    if manifest.get("split_digest") != recomputed["split_digest"]:
        raise IntegrityError(
            "split reproducibility FAILED: the split changed since the manifest was built "
            f"(stored {str(manifest.get('split_digest'))[:12]} != recomputed "
            f"{recomputed['split_digest'][:12]}) — row content changed or a row was added/removed."
        )

    # Locator binding on PORTABLE addresses: stored (sample_id, locator, cache_locator) must equal
    # the multiset re-derived from the actual rows. Locators are dataset-root-relative, so this
    # holds across clones (A-built artifact verifies against B's rows) while a redirected stored
    # locator — pointing a consumer at different data — is RED.
    def _locators(m: dict) -> list:
        return sorted(
            (str(r.get("sample_id", "")), str(r.get("locator", "")), str(r.get("cache_locator", "")))
            for r in m.get("rows", [])
        )

    if _locators(manifest) != _locators(recomputed):
        raise IntegrityError(
            "locator binding FAILED: a stored locator does not match the manifest rows it claims "
            "to describe — a storage locator was tampered/redirected."
        )


# --- CLI ----------------------------------------------------------------------------------------
def _read_manifest_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise IntegrityError(f"manifest {path!r} is empty")
    required = {"file_path", "taxonomy_v2_class"}
    missing = required - set(rows[0].keys())
    if missing:
        raise IntegrityError(f"manifest {path!r} missing columns: {', '.join(sorted(missing))}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track E versioned evaluation-integrity manifest (§8.1.5/§8.1.6)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build", help="build the versioned manifest")
    bp.add_argument("--manifest", required=True, help="input CSV: file_path,cache_path,taxonomy_v2_class")
    bp.add_argument("--source", required=True)
    bp.add_argument("--license", required=True, dest="license_")
    bp.add_argument("--label-authority", required=True, dest="label_authority")
    bp.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    bp.add_argument("--out", required=True)

    rp = sub.add_parser("report", help="per-category (real/synthetic/augmented) breakdown")
    rp.add_argument("--manifest-json", required=True, help="a versioned manifest produced by 'build'")

    vp = sub.add_parser("verify", help="tamper/drift check: re-derive from --manifest, compare to --manifest-json")
    vp.add_argument("--manifest", required=True, help="input CSV (source rows)")
    vp.add_argument("--manifest-json", required=True, help="a versioned manifest produced by 'build'")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "build":
            rows = _read_manifest_csv(args.manifest)
            manifest = build_versioned_manifest(
                rows,
                source=args.source,
                license_=args.license_,
                label_authority=args.label_authority,
                holdout_fraction=args.holdout_fraction,
            )
            Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"wrote {args.out} (manifest_digest {manifest['manifest_digest'][:12]})")
            return 0
        if args.cmd == "report":
            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            print(json.dumps(report_by_category(manifest), indent=2))
            return 0
        if args.cmd == "verify":
            rows = _read_manifest_csv(args.manifest)
            manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
            verify_manifest(rows, manifest)
            print("verify PASS: manifest matches the re-derived rows (no tamper/drift)")
            return 0
    except IntegrityError as exc:
        sys.stderr.write(f"[track-e] manifest integrity finding: {exc}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
