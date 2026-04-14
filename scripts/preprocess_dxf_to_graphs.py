#!/usr/bin/env python3
"""Preprocess DXF files into PyTorch graph tensors (.pt cache).

Converts each DXF file to node-feature + edge tensors and saves them as
individual .pt files so that training epochs no longer pay the ezdxf parse
cost (saves ~7 min/epoch → <30 s/epoch on 3k files).

Usage:
    # Cache training_v8 + augmented (3,069 files)
    python scripts/preprocess_dxf_to_graphs.py \
        --manifest data/manifests/training_v8_augmented_manifest.csv \
        --output-dir data/graph_cache

    # Cache full 24-class dataset (5,417 files)
    python scripts/preprocess_dxf_to_graphs.py \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output-dir data/graph_cache

    # Dry-run (process first 10 files)
    python scripts/preprocess_dxf_to_graphs.py \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output-dir data/graph_cache \
        --limit 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def file_hash(file_path: str) -> str:
    """MD5 hash of file path (fast, deterministic cache key)."""
    return hashlib.md5(file_path.encode()).hexdigest()


def convert_dxf(file_path: str, node_dim: int = 19, edge_dim: int = 7, with_text: bool = True):
    """Convert a DXF file to (x, edge_index, edge_attr, text) tensors.

    Reads the DXF file once and extracts both graph features and text content
    in a single ezdxf parse, avoiding the duplicate I/O cost of separate
    text extraction (B5.2a optimisation).

    Returns None on failure (empty graph, parse error, etc.).
    Returns (x, edge_index, edge_attr, text_str) on success.
    """
    try:
        import ezdxf
        from src.ml.train.dataset_2d import DXFDataset
        from src.ml.text_extractor import _extract_from_doc

        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()

        # Graph features
        ds = DXFDataset(root_dir=".", node_dim=node_dim, return_edge_attr=True)
        x, edge_index, edge_attr = ds._dxf_to_graph(
            msp, node_dim=node_dim, return_edge_attr=True
        )
        if x.size(0) == 0:
            return None

        # Text content — reuse the already-parsed doc (no second ezdxf.readfile)
        text_content = _extract_from_doc(doc) if with_text else ""

        return x, edge_index, edge_attr, text_content
    except Exception as e:
        logger.debug("Failed to convert %s: %s", file_path, e)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess DXF files to PyTorch graph cache (.pt)."
    )
    parser.add_argument(
        "--manifest", required=True,
        help="CSV manifest with file_path and taxonomy_v2_class columns."
    )
    parser.add_argument(
        "--output-dir", default="data/graph_cache",
        help="Directory to write .pt cache files (default: data/graph_cache)."
    )
    parser.add_argument(
        "--node-dim", type=int, default=19,
        help="Node feature dimension (default: 19)."
    )
    parser.add_argument(
        "--edge-dim", type=int, default=7,
        help="Edge feature dimension (default: 7)."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N files (0 = all, for dry-runs)."
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip files already in cache (default: True)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if cache file exists."
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="Skip text extraction (faster, omits 'text' field from cache)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = args.skip_existing and not args.force
    with_text = not args.no_text

    # Read manifest
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row.get("file_path", "").strip()
            label = (
                row.get("taxonomy_v2_class")
                or row.get("label_cn")
                or row.get("label")
                or ""
            ).strip()
            if fp and label:
                rows.append((fp, label))

    if args.limit > 0:
        rows = rows[: args.limit]

    logger.info("Processing %d files → %s", len(rows, ), output_dir)

    # Cache manifest: maps file_path → cache_path
    cache_manifest_path = output_dir / "cache_manifest.csv"
    existing_cache: dict[str, str] = {}
    if cache_manifest_path.exists():
        with open(cache_manifest_path, "r") as f:
            for row in csv.DictReader(f):
                existing_cache[row["file_path"]] = row["cache_path"]

    cache_manifest_rows = []
    t_start = time.time()
    ok, skipped, failed = 0, 0, 0

    for i, (fp, label) in enumerate(rows):
        key = file_hash(fp)
        cache_path = output_dir / f"{key}.pt"

        if skip_existing and fp in existing_cache and cache_path.exists():
            cache_manifest_rows.append({
                "file_path": fp,
                "cache_path": str(cache_path),
                "taxonomy_v2_class": label,
            })
            skipped += 1
            continue

        result = convert_dxf(fp, node_dim=args.node_dim, edge_dim=args.edge_dim, with_text=with_text)

        if result is None:
            failed += 1
            if (i + 1) % 100 == 0 or args.limit > 0:
                logger.info(
                    "[%d/%d] ok=%d skip=%d fail=%d  elapsed=%.0fs",
                    i + 1, len(rows), ok, skipped, failed,
                    time.time() - t_start,
                )
            continue

        x, edge_index, edge_attr, text_content = result
        torch.save(
            {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "label": label,
                "text": text_content,  # B5.2a: cached text avoids duplicate DXF reads
            },
            cache_path,
        )
        cache_manifest_rows.append({
            "file_path": fp,
            "cache_path": str(cache_path),
            "taxonomy_v2_class": label,
        })
        ok += 1

        if (i + 1) % 100 == 0 or args.limit > 0:
            elapsed = time.time() - t_start
            rate = (ok + failed) / max(elapsed, 1)
            remaining = (len(rows) - i - 1) / max(rate, 0.01)
            logger.info(
                "[%d/%d] ok=%d skip=%d fail=%d  %.1f f/s  ETA %.0fm",
                i + 1, len(rows), ok, skipped, failed,
                rate, remaining / 60,
            )

    # Write updated cache manifest
    with open(cache_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_path", "cache_path", "taxonomy_v2_class"]
        )
        writer.writeheader()
        writer.writerows(cache_manifest_rows)

    elapsed = time.time() - t_start
    logger.info(
        "Done. ok=%d  skipped=%d  failed=%d  total_time=%.0fs",
        ok, skipped, failed, elapsed,
    )
    logger.info("Cache manifest: %s (%d entries)", cache_manifest_path, len(cache_manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
