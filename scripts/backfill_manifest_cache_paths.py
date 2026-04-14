#!/usr/bin/env python3
"""Backfill missing cache_path values in a training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _hash_cache_path(cache_dir: Path, file_path: str) -> Path:
    digest = hashlib.md5(file_path.encode()).hexdigest()
    return cache_dir / f"{digest}.pt"


def _read_cache_map(cache_manifest: Path) -> dict[str, str]:
    cache_map: dict[str, str] = {}
    if not cache_manifest.exists() or not cache_manifest.is_file():
        return cache_map
    with open(cache_manifest, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            file_path = str(row.get("file_path", "")).strip()
            cache_path = str(row.get("cache_path", "")).strip()
            if file_path and cache_path:
                cache_map[file_path] = cache_path
    return cache_map


def backfill_manifest_cache_paths(
    *,
    manifest_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_manifest = cache_dir / "cache_manifest.csv"
    cache_map = _read_cache_map(cache_manifest)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Manifest has no header: {manifest_path}")
    if "cache_path" not in fieldnames:
        fieldnames.append("cache_path")

    filled = 0
    missing_before = 0
    for row in rows:
        cache_path = str(row.get("cache_path", "")).strip()
        if cache_path:
            continue
        missing_before += 1
        file_path = str(row.get("file_path", "")).strip()
        if not file_path:
            continue
        candidate = cache_map.get(file_path, "")
        if not candidate:
            hashed = _hash_cache_path(cache_dir, file_path)
            if hashed.exists():
                candidate = str(hashed)
        if candidate:
            row["cache_path"] = candidate
            filled += 1

    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    remaining = sum(1 for row in rows if not str(row.get("cache_path", "")).strip())
    return {
        "manifest": str(manifest_path),
        "cache_dir": str(cache_dir),
        "cache_manifest": str(cache_manifest),
        "rows": len(rows),
        "missing_before": missing_before,
        "filled": filled,
        "remaining": remaining,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill manifest cache_path values from cache_manifest.csv or hash lookup."
    )
    parser.add_argument("--manifest", required=True, help="Training manifest CSV to update in place.")
    parser.add_argument("--cache-dir", default="data/graph_cache", help="Directory containing .pt cache files and cache_manifest.csv.")
    parser.add_argument("--output-json", default="", help="Optional path to write a JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = backfill_manifest_cache_paths(
        manifest_path=Path(str(args.manifest)).expanduser(),
        cache_dir=Path(str(args.cache_dir)).expanduser(),
    )

    output_json = str(args.output_json or "").strip()
    if output_json:
        output_path = Path(output_json).expanduser()
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{json.dumps(result, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )

    print(f"Backfilled {result['filled']} cache_path entries", flush=True)
    if result["remaining"] > 0:
        print(
            f"FATAL: {result['remaining']} rows still missing cache_path after backfill",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
