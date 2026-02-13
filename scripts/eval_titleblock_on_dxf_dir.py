#!/usr/bin/env python3
"""Evaluate titleblock extraction/classification coverage on a DXF directory.

This script is intended for environments where manual DXF review is not feasible.
It produces:
- summary.json: aggregated counters + weak agreement rates vs filename weak labels
- predictions.csv: per-file titleblock + filename predictions
- errors.csv: parse/read failures

Artifacts are written to --output-dir (default: /tmp/titleblock_eval_<timestamp>).
The outputs include local file names/paths and are meant for local iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect_dxfs(root: Path, *, recursive: bool) -> List[Path]:
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.glob("*")
    files: List[Path] = []
    for path in candidates:
        try:
            if path.is_file() and path.suffix.lower() == ".dxf":
                files.append(path)
        except OSError:
            continue
    return sorted(files)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate TitleBlockClassifier coverage on a DXF directory."
    )
    parser.add_argument("--dxf-dir", required=True, help="DXF directory to scan.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for DXF files (default: false).",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory to write artifacts (default: /tmp/titleblock_eval_<ts>).",
    )
    parser.add_argument(
        "--synonyms-path",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms JSON path for label matching (default: template).",
    )
    parser.add_argument(
        "--true-label-min-confidence",
        type=float,
        default=0.8,
        help="Minimum filename weak-label confidence to accept as truth (default: 0.8).",
    )
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")

    files = _collect_dxfs(dxf_dir, recursive=bool(args.recursive))
    if not files:
        raise SystemExit(f"No DXF files found under: {dxf_dir}")

    random.seed(int(args.seed))
    random.shuffle(files)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if str(args.output_dir).strip()
        else Path("/tmp") / f"titleblock_eval_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep local artifacts out of git by default.
    try:
        (out_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    except Exception:
        pass

    synonyms_path = (
        str(args.synonyms_path).strip() or "data/knowledge/label_synonyms_template.json"
    )

    from src.ml.filename_classifier import FilenameClassifier
    from src.ml.titleblock_extractor import TitleBlockClassifier
    from src.utils.dxf_io import read_dxf_entities_from_bytes

    filename_clf = FilenameClassifier(synonyms_path=synonyms_path)
    titleblock_clf = TitleBlockClassifier(synonyms_path=synonyms_path)

    rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    status_counts: Counter[str] = Counter()
    filename_status_counts: Counter[str] = Counter()
    agree_counts: Counter[str] = Counter()
    titleblock_label_counts: Counter[str] = Counter()
    filename_label_counts: Counter[str] = Counter()

    strict_truth_total = 0
    strict_truth_agree = 0

    for path in files:
        rel_path = ""
        try:
            rel_path = str(path.relative_to(dxf_dir))
        except Exception:
            rel_path = path.name

        file_name = path.name
        raw_bytes: Optional[bytes] = None
        try:
            raw_bytes = path.read_bytes()
        except Exception as exc:
            error_rows.append(
                {
                    "relative_path": rel_path,
                    "file_name": file_name,
                    "error": f"read_error: {exc}",
                }
            )
            continue

        entities = None
        try:
            entities = read_dxf_entities_from_bytes(raw_bytes)
        except Exception as exc:
            error_rows.append(
                {
                    "relative_path": rel_path,
                    "file_name": file_name,
                    "error": f"parse_error: {exc}",
                }
            )
            continue

        titleblock_pred = titleblock_clf.predict(entities)
        filename_pred = filename_clf.predict(file_name)

        tb_status = str(titleblock_pred.get("status") or "").strip() or "unknown"
        status_counts[tb_status] += 1
        fn_status = str(filename_pred.get("status") or "").strip() or "unknown"
        filename_status_counts[fn_status] += 1

        tb_label = str(titleblock_pred.get("label") or "").strip() or ""
        fn_label = str(filename_pred.get("label") or "").strip() or ""
        tb_conf = _safe_float(titleblock_pred.get("confidence"), 0.0)
        fn_conf = _safe_float(filename_pred.get("confidence"), 0.0)

        if tb_label:
            titleblock_label_counts[tb_label] += 1
        if fn_label:
            filename_label_counts[fn_label] += 1

        agree = False
        if tb_label and fn_label and tb_label == fn_label:
            agree = True

        if tb_label and fn_label:
            agree_counts["both_present"] += 1
            agree_counts["agree"] += 1 if agree else 0
            agree_counts["disagree"] += 0 if agree else 1
        elif tb_label:
            agree_counts["titleblock_only"] += 1
        elif fn_label:
            agree_counts["filename_only"] += 1
        else:
            agree_counts["neither"] += 1

        # Strict truth: only accept filename weak labels above min confidence and stable status.
        strict_truth = bool(
            fn_label
            and fn_conf >= float(args.true_label_min_confidence)
            and fn_status in {"matched"}
        )
        if strict_truth:
            strict_truth_total += 1
            if agree:
                strict_truth_agree += 1

        title_info = titleblock_pred.get("title_block_info") or {}
        if not isinstance(title_info, dict):
            title_info = {}

        rows.append(
            {
                "relative_path": rel_path,
                "file_name": file_name,
                "titleblock_status": tb_status,
                "titleblock_label": tb_label or None,
                "titleblock_confidence": round(tb_conf, 4),
                "titleblock_part_name": title_info.get("part_name"),
                "titleblock_part_name_normalized": title_info.get(
                    "part_name_normalized"
                ),
                "titleblock_drawing_number": title_info.get("drawing_number"),
                "titleblock_material": title_info.get("material"),
                "titleblock_raw_texts_count": title_info.get("raw_texts_count"),
                "titleblock_region_entities_count": title_info.get(
                    "region_entities_count"
                ),
                "filename_status": fn_status,
                "filename_label": fn_label or None,
                "filename_confidence": round(fn_conf, 4),
                "filename_extracted_name": filename_pred.get("extracted_name"),
                "agree_with_filename": "Y" if agree else "N",
                "strict_truth": "Y" if strict_truth else "N",
            }
        )

    predictions_csv = out_dir / "predictions.csv"
    errors_csv = out_dir / "errors.csv"
    summary_json = out_dir / "summary.json"

    _write_csv(predictions_csv, rows)
    _write_csv(errors_csv, error_rows)

    both_present = int(agree_counts.get("both_present", 0))
    agree = int(agree_counts.get("agree", 0))
    agree_rate = float(agree) / float(both_present) if both_present else 0.0
    strict_agree_rate = (
        float(strict_truth_agree) / float(strict_truth_total)
        if strict_truth_total
        else 0.0
    )

    summary: Dict[str, Any] = {
        "status": "ok",
        "dxf_dir": str(dxf_dir),
        "files_total": len(files),
        "predictions_rows": len(rows),
        "errors_rows": len(error_rows),
        "synonyms_path": synonyms_path,
        "true_label_min_confidence": float(args.true_label_min_confidence),
        "counts": {
            "titleblock_status": dict(status_counts),
            "filename_status": dict(filename_status_counts),
            "agreement": dict(agree_counts),
        },
        "agreement": {
            "both_present": both_present,
            "agree": agree,
            "agree_rate": round(agree_rate, 6),
            "strict_truth_total": int(strict_truth_total),
            "strict_truth_agree": int(strict_truth_agree),
            "strict_truth_agree_rate": round(strict_agree_rate, 6),
        },
        "top_labels": {
            "titleblock": titleblock_label_counts.most_common(30),
            "filename": filename_label_counts.most_common(30),
        },
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "errors_csv": str(errors_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary.get("agreement", {}), ensure_ascii=False))
    print(f"wrote={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
