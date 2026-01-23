#!/usr/bin/env python3
"""Clean a DXF label manifest by merging or dropping low-frequency labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Clean DXF label manifest CSV.")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input manifest CSV (must include label_cn column)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output cleaned manifest CSV",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Minimum label frequency to keep (default: 3)",
    )
    parser.add_argument(
        "--other-label",
        default="other",
        help="Label used to replace low-frequency classes (default: other)",
    )
    parser.add_argument(
        "--drop-low",
        action="store_true",
        help="Drop low-frequency rows instead of mapping to other-label",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    rows = _load_rows(input_path)
    if not rows:
        raise SystemExit("No rows found in input manifest.")

    labels = Counter((row.get("label_cn") or "").strip() for row in rows)
    labels.pop("", None)
    keep_labels = {label for label, count in labels.items() if count >= args.min_count}

    cleaned_rows: list[dict[str, str]] = []
    replaced = 0
    dropped = 0
    for row in rows:
        label = (row.get("label_cn") or "").strip()
        if not label:
            continue
        if label in keep_labels:
            cleaned_rows.append(row)
            continue
        if args.drop_low:
            dropped += 1
            continue
        row = dict(row)
        row["label_cn"] = args.other_label
        cleaned_rows.append(row)
        replaced += 1

    fieldnames = list(rows[0].keys())
    _write_rows(output_path, fieldnames, cleaned_rows)

    print(f"rows_in={len(rows)}")
    print(f"rows_out={len(cleaned_rows)}")
    print(f"labels_in={len(labels)}")
    print(f"labels_kept={len(keep_labels)}")
    print(f"replaced={replaced}")
    print(f"dropped={dropped}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
