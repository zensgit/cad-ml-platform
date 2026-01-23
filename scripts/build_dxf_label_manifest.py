#!/usr/bin/env python3
"""Build a DXF label manifest from file names.

Extracts Chinese labels and version tags from DXF filenames and writes a CSV
for downstream Graph2D training.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_CN_PATTERN = re.compile(r"[\u4e00-\u9fff()\uFF08\uFF09]+")
_VERSION_PATTERN = re.compile(r"v\d+", re.IGNORECASE)
_COMPARISON_TERMS = {"\u6bd4\u8f83", "\u5bf9\u6bd4"}


def _extract_label(stem: str) -> str:
    matches = _CN_PATTERN.findall(stem)
    if not matches:
        return ""
    cleaned = [m for m in matches if m not in _COMPARISON_TERMS]
    if not cleaned:
        cleaned = matches
    return max(cleaned, key=len)


def _extract_version(stem: str) -> str:
    matches = _VERSION_PATTERN.findall(stem)
    if not matches:
        return ""
    return matches[-1].lower()


def _is_comparison(stem: str) -> bool:
    lower = stem.lower()
    return any(term in stem for term in _COMPARISON_TERMS) or " vs " in lower or "vs" in lower


def _iter_dxf_paths(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ["*.dxf", "*.DXF"]
    paths: list[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(sorted(input_dir.rglob(pattern)))
        else:
            paths.extend(sorted(input_dir.glob(pattern)))
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
        for path in _iter_dxf_paths(input_dir, recursive):
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
    parser = argparse.ArgumentParser(description="Build DXF label manifest CSV.")
    parser.add_argument(
        "--input-dir",
        action="append",
        help="Directory containing DXF files (repeatable)",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/MECH_DXF_LABEL_MANIFEST_20260123.csv",
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
        input_dirs = [Path("/Users/huazhou/Downloads/\u8bad\u7ec3\u56fe\u7eb8/\u8bad\u7ec3\u56fe\u7eb8_dxf")]
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
