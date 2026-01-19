#!/usr/bin/env python3
"""Build a DWG label manifest from file names.

Extracts Chinese labels and version tags from DWG filenames and writes a CSV
for downstream DXF conversion and rule generation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_CN_PATTERN = re.compile(r"[\u4e00-\u9fff()（）]+")
_VERSION_PATTERN = re.compile(r"v\d+", re.IGNORECASE)


def _extract_label(stem: str) -> str:
    matches = _CN_PATTERN.findall(stem)
    if not matches:
        return ""
    cleaned = [m for m in matches if m not in {"比较", "对比"}]
    if not cleaned:
        cleaned = matches
    # Use the longest match as primary label.
    return max(cleaned, key=len)


def _extract_version(stem: str) -> str:
    matches = _VERSION_PATTERN.findall(stem)
    if not matches:
        return ""
    return matches[-1].lower()


def _is_comparison(stem: str) -> bool:
    lower = stem.lower()
    return "比较" in stem or "对比" in stem or " vs " in lower or "vs" in lower


def _iter_dwg_paths(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ["*.dwg", "*.DWG"]
    paths: list[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(sorted(input_dir.rglob(pattern)))
        else:
            paths.extend(sorted(input_dir.glob(pattern)))
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _build_manifest(input_dirs: list[Path], recursive: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_dir in input_dirs:
        for path in _iter_dwg_paths(input_dir, recursive):
            stem = path.stem
            label = _extract_label(stem)
            version = _extract_version(stem)
            is_comparison = _is_comparison(stem)
            part_code = ""
            if label and label in stem:
                part_code = stem.split(label)[0].strip("_- ")
            rows.append(
                {
                    "file_name": path.name,
                    "stem": stem,
                    "label_cn": label,
                    "version": version,
                    "part_code": part_code,
                    "is_comparison": "1" if is_comparison else "0",
                    "relative_path": str(path.relative_to(input_dir)),
                    "source_dir": input_dir.name,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DWG label manifest CSV.")
    parser.add_argument(
        "--input-dir",
        action="append",
        help="Directory containing DWG files (repeatable)",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/MECH_DWG_LABEL_MANIFEST_20260119.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively",
    )
    args = parser.parse_args()

    if args.input_dir:
        input_dirs = [Path(p) for p in args.input_dir]
    else:
        input_dirs = [Path("/Users/huazhou/Downloads/训练图纸/训练图纸")]
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(str(input_dir))

    rows = _build_manifest(input_dirs, args.recursive)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "stem",
                "label_cn",
                "version",
                "part_code",
                "is_comparison",
                "relative_path",
                "source_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
