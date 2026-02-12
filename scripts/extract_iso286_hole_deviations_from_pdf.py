#!/usr/bin/env python3
"""Extract ISO 286 hole deviations from GB/T 1800.2-2020 PDF.

This parser targets GB/T tables and merges extracted values into
`data/knowledge/iso286_hole_deviations.json`.

Notes:
- Tables 2–5 use text parsing via pypdf.
- Tables 6–16 use pdfplumber table extraction (optional dependency).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader
import os

DEFAULT_OUTPUT = Path("data/knowledge/iso286_hole_deviations.json")

TABLE_ABC = {
    "name": "Table 2 A/B/C",
    "pages": [7, 8],  # 0-based page indices
    "start_markers": ["表   2", "表 2"],
    "stop_markers": ["表   3", "表 3"],
    "groups": [
        {"symbol": "A", "grades": [9, 10, 11, 12, 13]},
        {"symbol": "B", "grades": [8, 9, 10, 11, 12, 13]},
        {"symbol": "C", "grades": [8, 9, 10, 11, 12, 13]},
    ],
}

TABLE_CD_D_E = {
    "name": "Table 3 CD/D/E",
    "pages": [8, 9],
    "start_markers": ["表   3", "表 3"],
    "stop_markers": ["表   4", "表 4"],
    "groups": [
        {"symbol": "CD", "grades": [6, 7, 8, 9, 10]},
        {"symbol": "D", "grades": [6, 7, 8, 9, 10, 11, 12, 13]},
        {"symbol": "E", "grades": [5, 6, 7, 8, 9, 10]},
    ],
}

TABLE_EF_F = {
    "name": "Table 4 EF/F",
    "pages": [10],
    "start_markers": ["表   4", "表 4"],
    "stop_markers": ["表   5", "表 5"],
    "groups": [
        {"symbol": "EF", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
        {"symbol": "F", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
    ],
}

TABLE_FG_G = {
    "name": "Table 5 FG/G",
    "pages": [11],
    "start_markers": ["表   5", "表 5"],
    "stop_markers": ["表   6", "表 6"],
    "groups": [
        {"symbol": "FG", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
        {"symbol": "G", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
    ],
}

TABLE_H = {
    "name": "Table 6 H",
    "pages": [12],
    "start_markers": ["表   6", "表 6"],
    "stop_markers": ["表   7", "表 7"],
    "groups": [
        {"symbol": "H", "grades": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
    ],
}

TABLE_JS = {
    "name": "Table 7 JS",
    "pages": [13],
    "start_markers": ["表   7", "表 7"],
    "stop_markers": ["表8", "表 8"],
    "groups": [
        {"symbol": "JS", "grades": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
    ],
}

TABLE_J_K = {
    "name": "Table 8 J/K",
    "pages": [14],
    "start_markers": ["表8", "表 8"],
    "stop_markers": ["表   9", "表 9"],
    "groups": [
        {"symbol": "J", "grades": [6, 7, 8, 9]},
        {"symbol": "K", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
    ],
}

TABLE_M_N = {
    "name": "Table 9 M/N",
    "pages": [15],
    "start_markers": ["表  9", "表   9", "表 9"],
    "stop_markers": ["表   10", "表 10"],
    "groups": [
        {"symbol": "M", "grades": [3, 4, 5, 6, 7, 8, 9, 10]},
        {"symbol": "N", "grades": [3, 4, 5, 6, 7, 8, 9, 10, 11]},
    ],
}

TABLE_P = {
    "name": "Table 10 P",
    "pages": [16],
    "start_markers": ["表   10", "表 10"],
    "stop_markers": ["表   11", "表 11"],
    "groups": [{"symbol": "P", "grades": [3, 4, 5, 6, 7, 8, 9, 10]}],
}

TABLE_R = {
    "name": "Table 11 R",
    "pages": [17, 18],
    "start_markers": ["表  11", "表   11", "表 11"],
    "stop_markers": ["表   12", "表 12"],
    "groups": [{"symbol": "R", "grades": [3, 4, 5, 6, 7, 8, 9, 10]}],
}

TABLE_S = {
    "name": "Table 12 S",
    "pages": [19, 20],
    "start_markers": ["表   12", "表 12"],
    "stop_markers": ["表   13", "表 13"],
    "groups": [{"symbol": "S", "grades": [3, 4, 5, 6, 7, 8, 9, 10]}],
}

TABLE_T_U = {
    "name": "Table 13 T/U",
    "pages": [21, 22],
    "start_markers": ["表   13", "表 13"],
    "stop_markers": ["表   14", "表 14"],
    "groups": [
        {"symbol": "T", "grades": [5, 6, 7, 8]},
        {"symbol": "U", "grades": [5, 6, 7, 8, 9, 10]},
    ],
}

TABLE_V_X_Y = {
    "name": "Table 14 V/X/Y",
    "pages": [23],
    "start_markers": ["表   14", "表 14"],
    "stop_markers": ["表   15", "表 15"],
    "groups": [
        {"symbol": "V", "grades": [5, 6, 7, 8]},
        {"symbol": "X", "grades": [5, 6, 7, 8, 9, 10]},
        {"symbol": "Y", "grades": [6, 7, 8, 9, 10]},
    ],
}

TABLE_Z_ZA = {
    "name": "Table 15 Z/ZA",
    "pages": [24],
    "start_markers": ["表   15", "表 15"],
    "stop_markers": ["表   16", "表 16"],
    "groups": [
        {"symbol": "Z", "grades": [6, 7, 8, 9, 10, 11]},
        {"symbol": "ZA", "grades": [6, 7, 8, 9, 10, 11]},
    ],
}

TABLE_ZB_ZC = {
    "name": "Table 16 ZB/ZC",
    "pages": [25],
    "start_markers": ["表   16", "表 16"],
    "groups": [
        {"symbol": "ZB", "grades": [7, 8, 9, 10, 11]},
        {"symbol": "ZC", "grades": [7, 8, 9, 10, 11]},
    ],
}

TABLES_PYPDF = [
    TABLE_ABC,
    TABLE_CD_D_E,
    TABLE_EF_F,
    TABLE_FG_G,
]

TABLES_PDFPLUMBER = [
    TABLE_H,
    TABLE_JS,
    TABLE_J_K,
    TABLE_M_N,
    TABLE_P,
    TABLE_R,
    TABLE_S,
    TABLE_T_U,
    TABLE_V_X_Y,
    TABLE_Z_ZA,
    TABLE_ZB_ZC,
]


POSITIVE_SYMBOLS = {"A", "B", "C", "CD", "D", "E", "EF", "F", "FG", "G"}
NEGATIVE_SYMBOLS = {"S", "T", "U", "V", "X", "Y", "Z", "ZA", "ZB", "ZC"}


VALID_SIZE_UPPERS = {
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    20,
    24,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    90,
    100,
    110,
    120,
    140,
    160,
    180,
    200,
    225,
    250,
    280,
    315,
    355,
    400,
    450,
    500,
    560,
    630,
    800,
    1000,
    1250,
    1600,
    2000,
    2500,
    3150,
}


def _normalize_number_text(text: str, join_digits: bool = True) -> str:
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"([+-])\s+(?=\d)", r"\1", text)
    if join_digits:
        text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return text


def _extract_numbers(line: str, split_lines: bool = False) -> List[float]:
    if split_lines:
        values: List[float] = []
        for segment in str(line).splitlines():
            if not segment.strip():
                continue
            normalized = _normalize_number_text(segment, join_digits=True)
            matches = re.findall(r"[+-]?\d+(?:\.\d+)?", normalized)
            values.extend(float(m) for m in matches)
        return values
    normalized = _normalize_number_text(line, join_digits=True)
    matches = re.findall(r"[+-]?\d+(?:\.\d+)?", normalized)
    return [float(m) for m in matches]


def _extract_size_upper(line: str) -> Optional[float]:
    match = re.match(r"^\s*[—–]\s*(\d+(?:\s\d+)*)", line)
    if match:
        return float(match.group(1).replace(" ", ""))
    match = re.match(r"^\s*(\d+(?:\s\d+)*)\s+(\d+(?:\s\d+)*)", line)
    if match:
        high = match.group(2)
        return float(high.replace(" ", ""))
    return None


def _is_size_line(line: str) -> bool:
    if re.search(r"[+-]\s*\d", line):
        return _extract_size_upper(line) is not None
    if _extract_size_upper(line) is None:
        return False
    if re.search(r"[—–\-至]", line):
        return True
    return len(re.findall(r"\d+(?:\.\d+)?", line)) >= 2


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _has_pdfplumber() -> bool:
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        return False
    return True


def _choose_value(values: List[float], grades: List[int], prefer_grade: int) -> float:
    if not values:
        raise ValueError("No values supplied")
    if prefer_grade in grades:
        return values[grades.index(prefer_grade)]
    first = values[0]
    if all(abs(v - first) < 1e-6 for v in values):
        return first
    return first


def _parse_table(reader: PdfReader, table: dict, prefer_grade: int) -> Dict[str, List[List[float]]]:
    lines: List[str] = []
    stop_markers = table.get("stop_markers", [])
    start_markers = table.get("start_markers", [])
    started = not start_markers

    for page_idx in table["pages"]:
        text = reader.pages[page_idx].extract_text(extraction_mode="layout") or ""
        for line in text.splitlines():
            if not line.strip():
                continue
            if not started and any(marker in line for marker in start_markers):
                started = True
                continue
            if not started:
                continue
            if stop_markers and any(marker in line for marker in stop_markers):
                return _parse_table_lines(lines, table, prefer_grade)
            lines.append(line.rstrip())
    return _parse_table_lines(lines, table, prefer_grade)


def _parse_table_pdfplumber(pdf_path: Path, table: dict, prefer_grade: int) -> Dict[str, List[List[float]]]:
    import pdfplumber

    debug = os.getenv("ISO_PDF_DEBUG") == "1"
    results: Dict[str, List[List[float]]] = {g["symbol"]: [] for g in table["groups"]}
    group_sizes = [len(group["grades"]) for group in table["groups"]]
    group_offsets: List[int] = []
    offset = 0
    for size in group_sizes:
        group_offsets.append(offset)
        offset += size

    group_columns: Dict[str, tuple[int, int]] = {}
    for idx, group in enumerate(table["groups"]):
        grades = group["grades"]
        if prefer_grade in grades:
            grade_idx = grades.index(prefer_grade)
            grade_value = prefer_grade
        else:
            grade_idx = 0
            grade_value = grades[0]
        group_columns[group["symbol"]] = (2 + group_offsets[idx] + grade_idx, grade_value)

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in table["pages"]:
            table_data = pdf.pages[page_idx].extract_table()
            if not table_data:
                continue
            header_idx = None
            for i, row in enumerate(table_data):
                if not row or len(row) < 2:
                    continue
                if row[0] and row[1] and ("大" in row[0]) and ("至" in row[1]):
                    header_idx = i
                    break
            if header_idx is None:
                if debug:
                    print(f"{table['name']} page {page_idx}: header not found")
                continue
            for row in table_data[header_idx + 1 :]:
                if not row or len(row) < 2:
                    continue
                upper = row[1]
                if not upper:
                    continue
                upper_clean = str(upper).replace(" ", "")
                if not re.search(r"\d", upper_clean):
                    continue
                try:
                    size_upper = float(upper_clean)
                except ValueError:
                    continue
                if size_upper not in VALID_SIZE_UPPERS:
                    continue
                for symbol, (col, grade_value) in group_columns.items():
                    if col >= len(row):
                        continue
                    cell = row[col]
                    if not cell:
                        continue
                    cell_text = str(cell)
                    nums = _extract_numbers(cell_text, split_lines=True)
                    if not nums:
                        continue
                    if "±" in cell_text and len(nums) == 1:
                        value = nums[0]
                        if cell_text.strip().endswith(".") and grade_value <= 12:
                            value = value / 10.0
                        ei = -abs(value)
                    else:
                        ei = min(nums)
                    results[symbol].append([size_upper, ei])
    return results


def _parse_table_lines(lines: List[str], table: dict, prefer_grade: int) -> Dict[str, List[List[float]]]:
    groups = table["groups"]
    group_sizes = [len(group["grades"]) for group in groups]
    total_cols = sum(group_sizes)
    results: Dict[str, List[List[float]]] = {g["symbol"]: [] for g in groups}

    i = 0
    while i < len(lines):
        line = lines[i]
        if not _is_size_line(line):
            i += 1
            continue
        size_upper = _extract_size_upper(line)
        if size_upper is None:
            i += 1
            continue
        if size_upper not in VALID_SIZE_UPPERS:
            i += 1
            continue
        j = i + 1
        values: List[float] = []
        while j < len(lines):
            if j != i + 1 and _is_size_line(lines[j]):
                break
            values.extend(_extract_numbers(lines[j]))
            if len(values) >= total_cols:
                break
            j += 1
        if not values:
            i += 1
            continue
        drop = None
        for offset in range(len(group_sizes) + 1):
            if sum(group_sizes[offset:]) == len(values):
                drop = offset
                break
        if drop is None and len(values) >= total_cols:
            drop = 0
            values = values[:total_cols]
        if drop is None:
            i += 1
            continue

        cursor = 0
        for idx, group in enumerate(groups):
            if idx < drop:
                continue
            size = group_sizes[idx]
            group_vals = values[cursor : cursor + size]
            if not group_vals:
                cursor += size
                continue
            chosen = _choose_value(group_vals, group["grades"], prefer_grade)
            results[group["symbol"]].append([size_upper, chosen])
            cursor += size
        i = j
    return results


def _validate_series(symbol: str, rows: List[List[float]]) -> tuple[bool, str]:
    if len(rows) < 3:
        return False, "insufficient rows"
    size_uppers = [row[0] for row in rows]
    if size_uppers != sorted(size_uppers):
        return False, "size uppers not sorted"
    if any(size not in VALID_SIZE_UPPERS for size in size_uppers):
        return False, "invalid size upper"
    values = [row[1] for row in rows]
    if any(abs(val) > 100000 for val in values):
        return False, "deviation out of range"
    if symbol in POSITIVE_SYMBOLS and any(val < 0 for val in values):
        return False, "negative EI in positive symbol"
    if symbol in NEGATIVE_SYMBOLS and any(val > 0 for val in values):
        return False, "positive EI in negative symbol"
    # Basic monotonicity guard
    if symbol in POSITIVE_SYMBOLS:
        for prev, curr in zip(values, values[1:]):
            if curr < prev:
                return False, "non-monotonic EI for positive symbol"
    if symbol in NEGATIVE_SYMBOLS:
        for prev, curr in zip(values, values[1:]):
            if curr > prev:
                return False, "non-monotonic EI for negative symbol"
    return True, ""


def _merge_series(existing: List[List[float]], updates: List[List[float]]) -> List[List[float]]:
    merged: Dict[float, float] = {float(size): float(val) for size, val in existing}
    for size, val in updates:
        merged[float(size)] = float(val)
    return [[size, merged[size]] for size in sorted(merged.keys())]


def _merge_payload(
    path: Path,
    updates: Dict[str, List[List[float]]],
    prefer_grade: int,
) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"deviations": {}, "units": "um"}
    deviations = payload.get("deviations", {})
    for symbol, rows in updates.items():
        if rows:
            existing = deviations.get(symbol, [])
            deviations[symbol] = _merge_series(existing, rows)
    payload["deviations"] = deviations
    payload["preferred_grade"] = prefer_grade
    payload["source"] = (
        "ISO 286-2 (isofits data) + GB/T 1800.2-2020 Tables 2–16 (holes)"
    )
    payload["notes"] = (
        "Lower deviation (EI) per symbol and size upper bound. "
        "Values extracted from GB/T 1800.2-2020 Tables 2–16 when available; "
        "other symbols sourced from isofits or derived from shaft deviations."
    )
    return payload


def _build_deviations(
    pdf_path: Path,
    prefer_grade: int,
    allow_partial: bool,
) -> Dict[str, List[List[float]]]:
    pdfplumber_available = _has_pdfplumber()
    if TABLES_PDFPLUMBER and not pdfplumber_available:
        message = (
            "pdfplumber is required to parse Tables 6–16 (H–ZC). "
            "Install pdfplumber or rerun with --allow-partial to skip these tables."
        )
        if allow_partial:
            _warn(message)
        else:
            raise SystemExit(message)
    reader = PdfReader(str(pdf_path))
    merged: Dict[str, List[List[float]]] = {}
    for table in TABLES_PYPDF:
        updates = _parse_table(reader, table, prefer_grade)
        for symbol, rows in updates.items():
            if not rows:
                continue
            ok, reason = _validate_series(symbol, rows)
            if ok:
                merged[symbol] = rows
            else:
                _warn(f"Skipping {symbol} series: {reason}")
    if pdfplumber_available:
        for table in TABLES_PDFPLUMBER:
            updates = _parse_table_pdfplumber(pdf_path, table, prefer_grade)
            for symbol, rows in updates.items():
                if not rows:
                    continue
                ok, reason = _validate_series(symbol, rows)
                if ok:
                    merged[symbol] = rows
                else:
                    _warn(f"Skipping {symbol} series: {reason}")
    return merged


def _write_report_csv(rows: Dict[str, List[List[float]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "size_upper", "ei_um"])
        for symbol in sorted(rows.keys()):
            for size_upper, ei in rows[symbol]:
                writer.writerow([symbol, size_upper, ei])


def _diff_rows(
    base: Dict[str, List[List[float]]],
    other: Dict[str, List[List[float]]],
) -> List[Tuple[str, float, float, float, float]]:
    diff: List[Tuple[str, float, float, float, float]] = []
    for symbol in sorted(base.keys()):
        if symbol not in other:
            continue
        base_map = {size: ei for size, ei in base[symbol]}
        other_map = {size: ei for size, ei in other[symbol]}
        for size in sorted(set(base_map.keys()) & set(other_map.keys())):
            ei_base = base_map[size]
            ei_other = other_map[size]
            delta = ei_other - ei_base
            if abs(delta) > 0:
                diff.append((symbol, size, ei_base, ei_other, delta))
    return diff


def _write_diff_csv(rows: List[Tuple[str, float, float, float, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "size_upper", "ei_base_um", "ei_compare_um", "delta_um"])
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to GB/T 1800.2-2020 PDF")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prefer-grade", type=int, default=6)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow running without pdfplumber; skips Tables 6–16.",
    )
    parser.add_argument("--report", help="Optional CSV report output path")
    parser.add_argument("--compare-grade", type=int, help="Optional grade to compare against")
    parser.add_argument("--compare-report", help="Optional CSV diff output path")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    merged = _build_deviations(pdf_path, args.prefer_grade, args.allow_partial)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_payload(output_path, merged, args.prefer_grade)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        _write_report_csv(merged, Path(args.report))
    if args.compare_grade:
        compare_rows = _build_deviations(pdf_path, args.compare_grade, args.allow_partial)
        diff = _diff_rows(merged, compare_rows)
        compare_path = (
            Path(args.compare_report)
            if args.compare_report
            else Path(f"reports/ISO286_EI_DIFF_{args.prefer_grade}_vs_{args.compare_grade}.csv")
        )
        _write_diff_csv(diff, compare_path)
    symbols = ", ".join(sorted(merged.keys())) or "None"
    print(f"Updated {output_path} with symbols: {symbols}")


if __name__ == "__main__":
    main()
