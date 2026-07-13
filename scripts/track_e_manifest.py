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
    """The base every locator is expressed against.

    FAIL-CLOSED (review): if any input path is ABSOLUTE, an explicit ``root`` is REQUIRED. Root
    inference (e.g. a common-parent heuristic) is forbidden — mixing in a file from a sibling
    directory would silently WIDEN the inferred root upward (``/tmp/dataset`` + ``/tmp/outside``
    → root ``/tmp``), legitimizing out-of-dataset files instead of rejecting them. The trust
    boundary must be declared, never derived from the data it is supposed to contain.
    ``None`` only when every row path is already relative (repo-style manifests).
    """
    if root is not None:
        return Path(root)
    for r in rows:
        fp = str(r.get("file_path", ""))
        cache = str(r.get("cache_path", ""))
        if (fp and Path(fp).is_absolute()) or (cache and Path(cache).is_absolute()):
            raise IntegrityError(
                "rows contain absolute paths but no dataset root was given — an explicit root is "
                "required (containment cannot be inferred without widening the trust boundary)"
            )
    return None


def _relative_locator(fp: str, root: Optional[Path]) -> str:
    """Dataset-root-RELATIVE, NFC-normalized, POSIX-style locator — CONTAINED in the root.

    The sample's portable address inside the versioned dataset: it IS digested, must be identical
    on every fresh clone, and never carries a host prefix. A full relative path (not a basename),
    so same-named files in different directories cannot collide.

    FAIL-CLOSED containment (review): a file OUTSIDE the dataset root, an absolute locator, or any
    locator containing a ``..`` segment is REJECTED — a versioned dataset must be self-contained,
    and an escaping locator would let a manifest address data outside the dataset it claims to
    describe.
    """
    import unicodedata as _ud

    p = Path(fp)
    if p.is_absolute():
        if root is None:
            raise IntegrityError(
                f"absolute path {fp!r} requires an explicit dataset root (containment cannot be "
                "inferred without widening the trust boundary)"
            )
        # STRICT containment: resolve both sides (symlinks/.. collapsed) and require the file to
        # be inside the root — relpath()-style '..'-emitting fallbacks are forbidden.
        try:
            rel = p.resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise IntegrityError(
                f"locator escapes the dataset root: {fp!r} is outside {str(root)!r} — a versioned "
                "dataset must be self-contained"
            ) from exc
        rel = str(rel)
    else:
        rel = str(p)  # already relative -> validated below
    locator = _ud.normalize("NFC", Path(rel).as_posix())
    _validate_locator(locator, original=fp)
    return locator


def _validate_locator(locator: str, *, original: str = "") -> None:
    """Reject absolute or root-escaping locators (fail-closed containment)."""
    ref = f" (from {original!r})" if original and original != locator else ""
    if Path(locator).is_absolute() or (len(locator) > 1 and locator[1] == ":"):  # POSIX + drive
        raise IntegrityError(f"locator must be dataset-root-relative, got absolute {locator!r}{ref}")
    if locator == ".." or locator.startswith("../") or "/../" in locator or locator.endswith("/.."):
        raise IntegrityError(
            f"locator escapes the dataset root: {locator!r}{ref} — the file is outside the "
            "dataset root; a versioned dataset must be self-contained"
        )


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
    # Resolve/validate the trust boundary FIRST (fail-closed before any file I/O): absolute rows
    # with no explicit root are rejected here, before compute_split touches the filesystem.
    dataset_root = _dataset_root(rows, root)
    split = compute_split(rows, holdout_fraction=holdout_fraction, root=root)
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

    Checks (the digest self-check alone is NOT trusted — a re-digesting attacker defeats it, so
    every load-bearing field is INDEPENDENTLY re-derived from the rows):
      1. **Envelope self-consistency** — recompute ``manifest_digest`` over the STORED envelope and
         compare. Catches a naive single-field tamper. NOT sufficient on its own: an attacker who
         also recomputes the digest passes this check, which is why 2–4 re-derive from the rows.
      2. **Split drift** — re-derive the split from the actual ``rows`` and compare ``split_digest``.
      3. **Locator binding** — the stored ``(sample_id, locator, cache_locator)`` pairs must equal
         the pairs re-derived from the rows; a redirected/escaping/absolute locator is RED.
      4. **Provenance binding** — the stored ``provenance_complete`` / ``unknown_provenance_rows``
         must equal the values re-derived from the rows, so a re-digested flip of the provenance
         verdict (e.g. incomplete → complete, or a zeroed unknown-count) is RED.

    RESIDUAL (documented, not defeated): the free-text ``source`` / ``license`` /
    ``label_authority`` are external inputs, not row-derived — ``verify`` re-derives the manifest
    FROM the stored values, so a self-consistent rewrite of those three cannot be distinguished.
    Acceptable for a non-blocking dry-run; a signing layer is the Phase-B answer.

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

    # Containment of STORED locators: even a re-digested (self-consistent) manifest must never
    # carry an absolute or root-escaping locator — reject before any consumer could resolve it.
    for r in manifest.get("rows", []):
        _validate_locator(str(r.get("locator", "")))
        if r.get("cache_locator"):
            _validate_locator(str(r.get("cache_locator", "")))
    for q in manifest.get("quarantined", []):
        _validate_locator(str(q.get("locator", "")))

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

    # Provenance binding: provenance_complete / unknown_provenance_rows are ROW-DERIVED (built from
    # the same rows), so a re-digested manifest that flipped the provenance verdict to GREEN (or
    # zeroed the unknown count) disagrees with the re-derivation and is RED. Without this, the only
    # guard on these fields is the digest self-check, which a re-digesting attacker defeats.
    for field in ("provenance_complete", "unknown_provenance_rows"):
        if manifest.get(field) != recomputed[field]:
            raise IntegrityError(
                f"provenance binding FAILED: stored {field}={manifest.get(field)!r} != row-derived "
                f"{recomputed[field]!r} — the provenance verdict was tampered (re-digested)."
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
    bp.add_argument("--root", default=None,
                    help="dataset root every locator is expressed against; REQUIRED when any row "
                         "path is absolute (root inference is forbidden — absolute rows with no "
                         "--root fail closed); omit only when every row path is already relative")
    bp.add_argument("--out", required=True)

    rp = sub.add_parser("report", help="per-category (real/synthetic/augmented) breakdown")
    rp.add_argument("--manifest-json", required=True, help="a versioned manifest produced by 'build'")

    vp = sub.add_parser("verify", help="tamper/drift check: re-derive from --manifest, compare to --manifest-json")
    vp.add_argument("--manifest", required=True, help="input CSV (source rows)")
    vp.add_argument("--manifest-json", required=True, help="a versioned manifest produced by 'build'")
    vp.add_argument("--root", default=None, help="dataset root for locator re-derivation (see build --root)")

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
                root=Path(args.root) if args.root else None,
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
            verify_manifest(rows, manifest, root=Path(args.root) if args.root else None)
            print("verify PASS: manifest matches the re-derived rows (no tamper/drift)")
            return 0
    except IntegrityError as exc:
        sys.stderr.write(f"[track-e] manifest integrity finding: {exc}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
