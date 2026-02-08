#!/usr/bin/env python3
"""Build a DXF label manifest CSV.

Supports two labeling modes:
- filename: use FilenameClassifier + synonyms mapping (recommended for real DXF datasets)
- parent_dir: label from parent directory name (useful for synthetic datasets stored as label/xxx.dxf)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_CN_PATTERN = re.compile(r"[\u4e00-\u9fff()\uFF08\uFF09]+")
_VERSION_PATTERN = re.compile(r"v\d+", re.IGNORECASE)
_COMPARISON_TERMS = {"\u6bd4\u8f83", "\u5bf9\u6bd4"}

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _resolve_default_output_csv() -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    return f"reports/experiments/{stamp}/DXF_LABEL_MANIFEST_{stamp}.csv"


def _label_from_filename(path: Path, standardize: bool) -> dict[str, str]:
    from src.ml.filename_classifier import FilenameClassifier

    clf = FilenameClassifier()
    pred = clf.predict(path.name)
    extracted = (pred.get("extracted_name") or "").strip()
    matched = (pred.get("label") or "").strip()
    confidence = pred.get("confidence")
    match_type = (pred.get("match_type") or "").strip()
    status = (pred.get("status") or "").strip()

    label_cn = matched if (standardize and matched) else extracted
    if not label_cn:
        label_cn = extracted

    return {
        "label_cn": label_cn,
        "label_raw": extracted,
        "label_standard": matched,
        "label_confidence": f"{float(confidence or 0.0):.4f}",
        "label_match_type": match_type,
        "label_status": status,
        "label_source": "filename",
    }


def _label_from_parent_dir(path: Path, standardize: bool) -> dict[str, str]:
    raw = path.parent.name.strip()
    label_cn = raw
    label_standard = ""
    confidence = 1.0
    match_type = "dir"
    status = "dir_label"

    if standardize:
        try:
            from src.ml.filename_classifier import FilenameClassifier

            clf = FilenameClassifier()
            matched, conf, mt = clf.match_label(raw)
            if matched:
                label_cn = matched
                label_standard = matched
                confidence = float(conf)
                match_type = str(mt)
                status = "matched"
        except Exception:
            # Keep raw label on any import/matching error.
            pass

    return {
        "label_cn": label_cn,
        "label_raw": raw,
        "label_standard": label_standard,
        "label_confidence": f"{float(confidence):.4f}",
        "label_match_type": match_type,
        "label_status": status,
        "label_source": "parent_dir",
    }


def _build_manifest(
    input_dirs: list[Path],
    recursive: bool,
    label_mode: str,
    standardize: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for input_dir in input_dirs:
        for path in _iter_dxf_paths(input_dir, recursive):
            stem = path.stem
            label_payload: dict[str, str]
            if label_mode == "parent_dir":
                label_payload = _label_from_parent_dir(path, standardize=standardize)
            else:
                label_payload = _label_from_filename(path, standardize=standardize)

            label = label_payload.get("label_cn", "")
            version = _extract_version(stem)
            is_comparison = _is_comparison(stem)
            part_code = ""
            label_raw = label_payload.get("label_raw", "")
            if label_raw and label_raw in stem:
                part_code = stem.split(label_raw)[0].strip("_- ")
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
                    **label_payload,
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
        default="",
        help="CSV output path",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively",
    )
    parser.add_argument(
        "--label-mode",
        choices=["filename", "parent_dir"],
        default="filename",
        help="How to derive labels for label_cn column.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Do not map extracted labels through label_synonyms (keep raw).",
    )
    args = parser.parse_args()

    if args.input_dir:
        input_dirs = [Path(p) for p in args.input_dir]
    else:
        # Keep script runnable in a clean repo checkout.
        input_dirs = [ROOT / "data/synthetic_v2"]
        if not args.recursive:
            args.recursive = True

    output_csv = args.output_csv.strip() if args.output_csv else ""
    if not output_csv:
        output_csv = _resolve_default_output_csv()

    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(str(input_dir))

    rows = _build_manifest(
        input_dirs,
        args.recursive,
        label_mode=args.label_mode,
        standardize=not args.no_standardize,
    )
    out_path = Path(args.output_csv)
    if not args.output_csv:
        out_path = Path(output_csv)
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
                "label_raw",
                "label_standard",
                "label_confidence",
                "label_match_type",
                "label_status",
                "label_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
