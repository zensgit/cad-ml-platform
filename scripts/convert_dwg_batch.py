#!/usr/bin/env python3
"""Batch convert DWG files to DXF using the configured converter."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.core.dedupcad_precision.cad_pipeline import convert_dwg_to_dxf


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch convert DWG to DXF.")
    parser.add_argument(
        "--input-dir",
        action="append",
        help="Directory containing DWG files (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/huazhou/Downloads/训练图纸/训练图纸_dxf",
        help="Output directory for DXF files",
    )
    parser.add_argument(
        "--log-csv",
        default="reports/MECH_DWG_TO_DXF_LOG_20260119.csv",
        help="CSV log output path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.input_dir:
        input_dirs = [Path(p) for p in args.input_dir]
    else:
        input_dirs = [Path("/Users/huazhou/Downloads/训练图纸/训练图纸")]
    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(str(input_dir))
        for dwg_path in _iter_dwg_paths(input_dir, args.recursive):
            out_path = output_dir / (dwg_path.stem + ".dxf")
            status = "ok"
            error = ""
            try:
                convert_dwg_to_dxf(dwg_path, out_path)
            except Exception as exc:
                status = "error"
                error = str(exc)
            rows.append(
                {
                    "file_name": dwg_path.name,
                    "output_path": str(out_path),
                    "status": status,
                    "error": error,
                    "relative_path": str(dwg_path.relative_to(input_dir)),
                    "source_dir": input_dir.name,
                }
            )

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "output_path",
                "status",
                "error",
                "relative_path",
                "source_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"Converted {ok}/{len(rows)} files. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
