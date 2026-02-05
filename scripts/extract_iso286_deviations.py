#!/usr/bin/env python3
"""Extract ISO 286 deviations tables from GB/T 1800.2 PDF.

This script attempts to parse hole/shaft deviation tables and produce a JSON
artifact that can be used for tolerance lookups. It is best-effort and keeps
raw table metadata to aid manual validation.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

KNOWN_HOLE_SYMBOLS = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "JS",
    "K",
    "M",
    "N",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "Z",
    "ZA",
    "ZB",
    "ZC",
    "CD",
    "EF",
    "FG",
}

KNOWN_SHAFT_SYMBOLS = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "j",
    "js",
    "k",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
    "y",
    "z",
    "za",
    "zb",
    "zc",
}


def _normalize_cell(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip()


def _parse_number(value: str) -> Optional[float]:
    if not value:
        return None
    value = value.replace("—", "").replace("–", "").replace("-", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def _parse_deviation_cell(cell: str) -> Optional[Tuple[float, float]]:
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", cell)
    if not numbers:
        return None
    if len(numbers) == 1:
        val = float(numbers[0])
        return val, val
    return float(numbers[0]), float(numbers[1])


def _extract_ascii_letters(value: str) -> str:
    return "".join(re.findall(r"[A-Za-z]+", value))


def _normalize_symbol(symbol: str, kind: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    tokens = symbol.split()
    if len(tokens) > 1:
        symbol = tokens[-1]
    symbol = _extract_ascii_letters(symbol)
    if not symbol:
        return None
    if kind == "holes":
        upper = symbol.upper()
        if upper in KNOWN_HOLE_SYMBOLS:
            return upper
        if len(symbol) > 1 and symbol[-1].islower():
            trimmed = symbol[:-1].upper()
            if trimmed in KNOWN_HOLE_SYMBOLS:
                return trimmed
        if len(symbol) > 1 and symbol[0].islower():
            trimmed = symbol[1:].upper()
            if trimmed in KNOWN_HOLE_SYMBOLS:
                return trimmed
        return upper
    if kind == "shafts":
        lower = symbol.lower()
        if lower in KNOWN_SHAFT_SYMBOLS:
            return lower
        if len(lower) > 1:
            trimmed = lower[:-1]
            if trimmed in KNOWN_SHAFT_SYMBOLS:
                return trimmed
            trimmed = lower[1:]
            if trimmed in KNOWN_SHAFT_SYMBOLS:
                return trimmed
        return lower

    return symbol


def _build_column_labels(
    header1: List[Any],
    header2: List[Any],
    kind: Optional[str],
) -> List[Optional[str]]:
    group_labels: List[Optional[str]] = []
    current: Optional[str] = None
    for cell in header1[2:]:
        label = _normalize_cell(cell)
        if label:
            current = label
        group_labels.append(current)

    grade_labels = [_normalize_cell(cell) for cell in header2[2:]]
    col_labels: List[Optional[str]] = []
    for group, grade in zip(group_labels, grade_labels):
        if not group or not grade:
            col_labels.append(None)
        else:
            normalized_group = _normalize_symbol(group, kind)
            if not normalized_group:
                col_labels.append(None)
            else:
                col_labels.append(f"{normalized_group}{grade}")
    return col_labels


def _extract_tables(page) -> List[List[List[Any]]]:
    return page.extract_tables(
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 5,
            "join_tolerance": 5,
        }
    )


def parse_pdf(pdf_path: Path) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now().isoformat(),
        "units": "um",
        "tables": [],
        "holes": {},
        "shafts": {},
        "warnings": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            kind = None
            if "孔的极限偏差" in text:
                kind = "holes"
            elif "轴的极限偏差" in text:
                kind = "shafts"

            tables = _extract_tables(page)
            if not tables:
                continue

            title = None
            for line in text.splitlines():
                if line.strip().startswith("表"):
                    title = line.strip()
                    break

            for table in tables:
                if len(table) < 3:
                    continue
                header1 = table[0]
                header2 = table[1]
                col_labels = _build_column_labels(header1, header2, kind)

                rows: List[Dict[str, Any]] = []
                for row in table[2:]:
                    if not row or len(row) < 3:
                        continue
                    lo_raw = _normalize_cell(row[0])
                    hi_raw = _normalize_cell(row[1])
                    lo = _parse_number(lo_raw)
                    hi = _parse_number(hi_raw)
                    if lo is None and hi is None:
                        continue

                    values: Dict[str, Dict[str, float]] = {}
                    for label, cell in zip(col_labels, row[2:]):
                        if not label:
                            continue
                        cell_text = _normalize_cell(cell)
                        if not cell_text:
                            continue
                        parsed = _parse_deviation_cell(cell_text)
                        if parsed is None:
                            continue
                        upper, lower = parsed
                        values[label] = {"upper": upper, "lower": lower}

                        if kind in ("holes", "shafts") and hi is not None:
                            collection = output[kind]
                            collection.setdefault(label, []).append([hi, lower, upper])

                    if values:
                        rows.append(
                            {
                                "size_min": lo,
                                "size_max": hi,
                                "values": values,
                            }
                        )

                output["tables"].append(
                    {
                        "page": page_index,
                        "title": title,
                        "kind": kind,
                        "columns": [c for c in col_labels if c],
                        "rows": rows,
                    }
                )

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path, help="Path to GB/T 1800.2 PDF")
    parser.add_argument(
        "--out",
        default=Path("data/knowledge/iso286_deviations.json"),
        type=Path,
        help="Output JSON path",
    )
    args = parser.parse_args()

    data = parse_pdf(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
