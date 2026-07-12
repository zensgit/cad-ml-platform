#!/usr/bin/env python3
"""Track E slice-2: versioned evaluation-integrity manifest + real/synthetic/augmented reporting.

Builds on slice-1 (``scripts/track_e_eval_integrity.py``): PRODUCT_STRATEGY.md §8.1.5 requires
separate reporting for real, synthetic, and augmented data, and §8.1.6 requires a versioned
manifest carrying source, license, provenance, family, hash, split, and label authority. Slice-1
already delivers the leakage-safe split (content-hash + normalized-family, conflict quarantine,
fail-closed unreadable content, deterministic digest); this module REUSES those primitives rather
than re-deriving them, so a manifest built here can never drift from the split slice-1 (and the
L3 gate behind it) already trusts.

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


def _enrich_rows(
    rows: List[dict],
    split: dict,
    *,
    source: str,
    license_: str,
    label_authority: str,
) -> List[dict]:
    """Build one §8.1.6-complete record per row that survived slice-1's ``compute_split``.

    Rows slice-1 quarantined (missing/unreadable content, label conflict, missing field) are
    silently absent from ``split["assignment"]`` and therefore excluded here too — fail-closed
    behaviour is inherited, not re-implemented. The ``content_hash`` is the SAME snapshot
    ``compute_split`` computed to build the split (``split["content_hashes"]``) — NOT a second read
    — so the manifest's hash can never disagree with the split it records (review P2: a file
    changing between two independent reads could otherwise let identical content straddle).
    """
    content_hashes = split["content_hashes"]
    enriched: List[dict] = []
    for row in rows:
        key = str(row.get("file_path", ""))
        if key not in split["assignment"]:
            continue  # quarantined upstream by compute_split; never enters the manifest
        enriched.append(
            {
                "file_path": key,
                "cache_path": str(row.get("cache_path", "")),
                "taxonomy_v2_class": str(row.get("taxonomy_v2_class", "")).strip(),
                "family": _family_key(row),
                "content_hash": content_hashes[key],   # the authoritative split-time snapshot
                "split": split["assignment"][key],
                "category": categorize(row),
                "source": source,
                "license": license_,
                "label_authority": label_authority,
            }
        )
    return enriched


def _manifest_digest(enriched_rows: Iterable[dict]) -> str:
    """sha256 over the sorted enriched rows — deterministic and order-independent.

    Each row is first canonicalized to a sort-keyed JSON string (so field-order never matters),
    then the list of those strings is sorted (so row order never matters) before hashing.
    """
    canonical = sorted(
        json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for r in enriched_rows
    )
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
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

    Quarantined rows (§8.1.2 conflicts, unreadable content, missing fields) are excluded from
    ``rows`` and surfaced separately in ``quarantined`` for audit, never silently dropped.
    """
    rows = list(rows)
    split = compute_split(rows, holdout_fraction=holdout_fraction, root=root)
    enriched = _enrich_rows(
        rows, split, source=source, license_=license_, label_authority=label_authority
    )
    unknown_provenance = sum(1 for r in enriched if r["category"] == "unknown")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "license": license_,
        "label_authority": label_authority,
        "holdout_fraction": holdout_fraction,
        # §8.1.5 provenance completeness: any row of unknown provenance means the manifest is NOT a
        # clean, fully-attributed dataset — a downstream evaluation must NOT treat it as complete
        # (keeps the L3 exit condition blocked until provenance is resolved).
        "provenance_complete": unknown_provenance == 0,
        "unknown_provenance_rows": unknown_provenance,
        "rows": enriched,
        "quarantined": list(split["quarantined"]),
        "manifest_digest": _manifest_digest(enriched),
        "split_digest": split_digest(split),
    }


def report_by_category(manifest: dict) -> dict:
    """Per-category (§8.1.5) breakdown: counts per category, per (category × split), and per
    (category × taxonomy_v2_class). Every sub-breakdown sums back to the total row count."""
    rows = manifest.get("rows", [])
    by_category: Dict[str, int] = {c: 0 for c in CATEGORIES}
    by_category_split: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}
    by_category_class: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}

    for row in rows:
        cat = row.get("category", "real")
        by_category[cat] = by_category.get(cat, 0) + 1

        split = row.get("split", "")
        cat_split = by_category_split.setdefault(cat, {})
        cat_split[split] = cat_split.get(split, 0) + 1

        cls = row.get("taxonomy_v2_class", "")
        cat_class = by_category_class.setdefault(cat, {})
        cat_class[cls] = cat_class.get(cls, 0) + 1

    return {
        "total_rows": len(rows),
        "by_category": by_category,
        "by_category_split": by_category_split,
        "by_category_taxonomy_v2_class": by_category_class,
    }


def verify_manifest(rows: Iterable[dict], manifest: dict, *, root: Optional[Path] = None) -> None:
    """Re-derive the versioned manifest from ``rows`` and confirm both digests match. RAISES
    ``IntegrityError`` on any drift — tamper, content change, or an added/removed/moved row.

    Dry-run posture (non-blocking), mirroring slice-1's ``verify_reproducible``: callers decide
    what to do with the raised error (e.g. print-and-exit-nonzero in a CI check), this function
    never exits or prints on its own.
    """
    holdout_fraction = manifest.get("holdout_fraction", DEFAULT_HOLDOUT_FRACTION)
    recomputed = build_versioned_manifest(
        rows,
        source=str(manifest.get("source", "")),
        license_=str(manifest.get("license", "")),
        label_authority=str(manifest.get("label_authority", "")),
        holdout_fraction=holdout_fraction,
        root=root,
    )
    stored_manifest_digest = manifest.get("manifest_digest")
    stored_split_digest = manifest.get("split_digest")
    if (
        stored_manifest_digest != recomputed["manifest_digest"]
        or stored_split_digest != recomputed["split_digest"]
    ):
        raise IntegrityError(
            "reproducibility check FAILED: the manifest changed since it was built "
            f"(manifest_digest stored {str(stored_manifest_digest)[:12]} != recomputed "
            f"{recomputed['manifest_digest'][:12]}; split_digest stored "
            f"{str(stored_split_digest)[:12]} != recomputed {recomputed['split_digest'][:12]}). "
            "A manifest was tampered, row content changed, or a row was added/removed."
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
