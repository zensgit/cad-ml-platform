#!/usr/bin/env python3
"""Normalize DXF label manifest by mapping labels to canonical buckets."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.label_normalization import DXF_LABEL_BUCKET_MAP  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize DXF label manifest CSV.")
    parser.add_argument("--input-csv", required=True, help="Input manifest CSV")
    parser.add_argument("--output-csv", required=True, help="Output normalized CSV")
    parser.add_argument(
        "--default-label",
        default="other",
        help="Fallback label for unmapped entries",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when encountering unmapped labels",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    rows = _load_rows(input_path)
    if not rows:
        raise SystemExit("No rows found in input manifest.")

    counts = Counter()
    normalized_rows: list[dict[str, str]] = []
    unmapped = set()

    for row in rows:
        label = (row.get("label_cn") or "").strip()
        if not label:
            continue
        mapped = DXF_LABEL_BUCKET_MAP.get(label)
        if mapped is None:
            unmapped.add(label)
            if args.strict:
                raise SystemExit(f"Unmapped label: {label}")
            mapped = args.default_label
        new_row = dict(row)
        new_row["label_raw"] = label
        new_row["label_cn"] = mapped
        normalized_rows.append(new_row)
        counts[mapped] += 1

    fieldnames = list(rows[0].keys())
    if "label_raw" not in fieldnames:
        fieldnames.append("label_raw")

    _write_rows(output_path, fieldnames, normalized_rows)

    print(f"rows_in={len(rows)}")
    print(f"rows_out={len(normalized_rows)}")
    print(f"labels_out={len(counts)}")
    print(f"output={output_path}")
    for label, count in counts.most_common():
        print(f"{label},{count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
