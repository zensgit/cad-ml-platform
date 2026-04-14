#!/usr/bin/env python3
"""Append human-reviewed low-confidence samples to training manifest (B5.4b).

Reads the LowConfidenceQueue CSV, filters entries that have a reviewed_label,
deduplicates against the existing manifest, and writes a new manifest that
includes the newly confirmed samples.

This is the bridge between the active-learning review loop and the next
incremental training run:

  data/review_queue/low_conf.csv   →  this script  →  unified_manifest_v3.csv
                                                             ↓
                             finetune_graph2d_v2_augmented.py (→ v4 model)

Usage:
    # Dry-run (shows what would be appended, no file written)
    python scripts/append_reviewed_to_manifest.py \
        --queue data/review_queue/low_conf.csv \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output data/manifests/unified_manifest_v3.csv \
        --dry-run

    # Write new manifest
    python scripts/append_reviewed_to_manifest.py \
        --queue data/review_queue/low_conf.csv \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output data/manifests/unified_manifest_v3.csv

    # Only append corrections (reviewed_label ≠ predicted_class)
    python scripts/append_reviewed_to_manifest.py \
        --queue data/review_queue/low_conf.csv \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output data/manifests/unified_manifest_v3.csv \
        --corrections-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _read_manifest(path: str) -> list[dict]:
    """Read a training manifest CSV. Returns list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _manifest_file_paths(rows: list[dict]) -> set[str]:
    """Extract the set of file_path values from a manifest."""
    return {r.get("file_path", "").strip() for r in rows if r.get("file_path")}


def _write_manifest(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to a CSV manifest with the given fieldnames."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Queue helpers ─────────────────────────────────────────────────────────────

def _read_queue(path: str) -> list[dict]:
    """Read the LowConfidenceQueue CSV. Returns list of row dicts."""
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _reviewed_rows(queue_rows: list[dict], corrections_only: bool = False) -> list[dict]:
    """Return queue rows that have a non-empty reviewed_label.

    Args:
        queue_rows: All rows from the queue CSV.
        corrections_only: If True, only include rows where reviewed_label
            differs from predicted_class (genuine corrections).
    """
    result = []
    for row in queue_rows:
        label = row.get("reviewed_label", "").strip()
        if not label:
            continue
        if corrections_only:
            predicted = row.get("predicted_class", "").strip()
            if label == predicted:
                continue
        result.append(row)
    return result


# ── DXF file discovery ────────────────────────────────────────────────────────

def _find_dxf_by_hash(
    file_hash: str,
    search_roots: list[str],
    hash_len: int = 12,
) -> Optional[str]:
    """Search for a DXF file whose MD5 prefix matches file_hash.

    This is best-effort: it only works when the original DXF files are
    still available in the expected locations.

    Returns the first matching absolute path, or None if not found.
    """
    for root in search_roots:
        for dxf_path in Path(root).rglob("*.dxf"):
            candidate_hash = hashlib.md5(dxf_path.read_bytes()).hexdigest()[:hash_len]
            if candidate_hash == file_hash[:hash_len]:
                return str(dxf_path)
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append reviewed low-conf samples to training manifest."
    )
    parser.add_argument(
        "--queue",
        default="data/review_queue/low_conf.csv",
        help="LowConfidenceQueue CSV (default: data/review_queue/low_conf.csv).",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Existing training manifest CSV to extend.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest path (can be same as --manifest to extend in-place).",
    )
    parser.add_argument(
        "--dxf-roots",
        nargs="*",
        default=[],
        help="Root directories to search for original DXF files by hash. "
             "Required when queue rows reference files by hash only.",
    )
    parser.add_argument(
        "--corrections-only",
        action="store_true",
        help="Only append rows where reviewed_label ≠ predicted_class.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be appended without writing any files.",
    )
    parser.add_argument(
        "--label-column",
        default="taxonomy_v2_class",
        help="Label column name in the manifest (default: taxonomy_v2_class).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Read existing manifest
    existing_rows = _read_manifest(args.manifest)
    existing_paths = _manifest_file_paths(existing_rows)
    label_col = args.label_column

    # Infer manifest fieldnames
    if existing_rows:
        fieldnames = list(existing_rows[0].keys())
        if label_col not in fieldnames:
            fieldnames.append(label_col)
    else:
        fieldnames = ["file_path", label_col]

    logger.info("Existing manifest: %d rows  (%s)", len(existing_rows), args.manifest)

    # Read queue
    queue_rows = _read_queue(args.queue)
    logger.info("Queue: %d total rows  (%s)", len(queue_rows), args.queue)

    reviewed = _reviewed_rows(queue_rows, corrections_only=args.corrections_only)
    logger.info(
        "Reviewed rows: %d  (corrections_only=%s)", len(reviewed), args.corrections_only
    )

    # Build new rows
    new_rows: list[dict] = []
    skipped_no_path = 0
    skipped_duplicate = 0

    for qrow in reviewed:
        file_hash = qrow.get("file_hash", "").strip()
        reviewed_label = qrow.get("reviewed_label", "").strip()
        filename = qrow.get("filename", "").strip()

        # Try to resolve file_path from hash
        file_path = _find_dxf_by_hash(file_hash, args.dxf_roots) if args.dxf_roots else None

        if not file_path:
            # Try filename as relative path (in case it was stored that way)
            if filename and Path(filename).exists():
                file_path = str(Path(filename).resolve())
            else:
                logger.warning(
                    "Cannot resolve DXF path for hash=%s filename=%s — skipped",
                    file_hash, filename,
                )
                skipped_no_path += 1
                continue

        if file_path in existing_paths:
            logger.debug("Duplicate path already in manifest: %s — skipped", file_path)
            skipped_duplicate += 1
            continue

        new_row = {f: "" for f in fieldnames}
        new_row["file_path"] = file_path
        new_row[label_col] = reviewed_label
        new_rows.append(new_row)
        existing_paths.add(file_path)

        logger.info("  + %s  →  %s", file_path, reviewed_label)

    logger.info(
        "Summary: +%d new rows  (skipped: %d no-path  %d duplicates)",
        len(new_rows), skipped_no_path, skipped_duplicate,
    )

    if args.dry_run:
        logger.info("Dry-run: no file written.")
        return 0

    if not new_rows:
        logger.info("Nothing to append — output not written.")
        return 0

    all_rows = existing_rows + new_rows
    _write_manifest(args.output, all_rows, fieldnames)
    logger.info("Written %d rows → %s", len(all_rows), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
