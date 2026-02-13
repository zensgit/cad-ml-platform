#!/usr/bin/env python3
"""Diagnose Graph2D classifier behavior on a DXF directory.

This is intended for environments where manual DXF review is not feasible.
It reports:
- label distribution
- confidence distribution
- (optional) accuracy when directory structure encodes labels (parent folder name)
- (optional) accuracy when a manifest CSV provides labels
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_synonyms(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {
        str(k): [str(v) for v in values]
        for k, values in data.items()
        if isinstance(values, list)
    }


def _build_alias_map(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    for label, values in synonyms.items():
        alias[label.lower()] = label
        for value in values:
            value = str(value or "").strip()
            if value:
                alias[value.lower()] = label
    return alias


def _canonical(label: str, alias_map: Dict[str, str]) -> str:
    cleaned = str(label or "").strip()
    if not cleaned:
        return ""
    return alias_map.get(cleaned.lower(), cleaned)


def _collect_dxfs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.dxf") if p.is_file()]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_counts(counter: Counter[str], topn: int = 20) -> List[Tuple[str, int]]:
    return [(k, int(v)) for k, v in counter.most_common(topn)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Graph2D classifier on a DXF directory."
    )
    parser.add_argument(
        "--dxf-dir",
        default="data/synthetic_v2",
        help="Directory containing DXF files (can be nested).",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv(
            "GRAPH2D_MODEL_PATH",
            "models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth",
        ),
        help="Graph2D checkpoint path.",
    )
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=f"reports/experiments/{time.strftime('%Y%m%d')}/graph2d_diagnose",
        help="Directory to write CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--labels-from-parent-dir",
        action="store_true",
        help="Treat parent directory name as ground-truth label (for synthetic datasets).",
    )
    parser.add_argument(
        "--manifest-csv",
        default="",
        help="Optional manifest CSV used as ground-truth labels (expects file_name + label_cn).",
    )
    parser.add_argument(
        "--labels-from-filename",
        action="store_true",
        help="Treat FilenameClassifier prediction as ground-truth label (weak supervision).",
    )
    parser.add_argument(
        "--true-label-min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence to accept a weak-supervised true label (default: 0.8).",
    )
    parser.add_argument(
        "--synonyms-json",
        default="data/knowledge/label_synonyms_template.json",
        help="Synonyms JSON path used to canonicalize labels (default: template).",
    )
    parser.add_argument(
        "--include-abs-paths",
        action="store_true",
        help="Include absolute input directory path in summary output (default: false).",
    )
    args = parser.parse_args()

    dxf_dir = Path(args.dxf_dir)
    if not dxf_dir.exists():
        raise SystemExit(f"DXF dir not found: {dxf_dir}")

    files = _collect_dxfs(dxf_dir)
    if not files:
        raise SystemExit(f"No DXF files found under: {dxf_dir}")

    random.seed(int(args.seed))
    random.shuffle(files)
    files = files[: int(args.max_files)]

    truth_modes = sum(
        [
            1 if bool(args.labels_from_parent_dir) else 0,
            1 if bool(args.labels_from_filename) else 0,
            1 if str(args.manifest_csv or "").strip() else 0,
        ]
    )
    if truth_modes > 1:
        raise SystemExit(
            "Choose only one ground-truth mode: "
            "--labels-from-parent-dir OR --labels-from-filename OR --manifest-csv"
        )

    from src.ml.vision_2d import Graph2DClassifier

    clf = Graph2DClassifier(model_path=str(args.model_path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-file outputs contain file names; keep untracked by default.
    (out_dir / ".gitignore").write_text("predictions.csv\n", encoding="utf-8")

    synonyms_path = ROOT / str(args.synonyms_json)
    synonyms = _load_synonyms(synonyms_path)
    alias_map = _build_alias_map(synonyms)

    filename_classifier = None
    if bool(args.labels_from_filename):
        from src.ml.filename_classifier import FilenameClassifier

        filename_classifier = FilenameClassifier(synonyms_path=str(synonyms_path))

    manifest_labels_by_relpath: Dict[str, Dict[str, str]] = {}
    manifest_labels_by_name: Dict[str, Dict[str, str]] = {}
    manifest_csv = Path(str(args.manifest_csv).strip()) if str(args.manifest_csv).strip() else None
    if manifest_csv is not None and manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                file_name = str(row.get("file_name") or "").strip()
                label_cn = str(row.get("label_cn") or "").strip()
                if not file_name or not label_cn:
                    continue
                rel_path = str(row.get("relative_path") or "").strip()
                if rel_path:
                    manifest_labels_by_relpath[rel_path] = row
                if file_name not in manifest_labels_by_name:
                    manifest_labels_by_name[file_name] = row

    rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    pred_canon_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    confusion_counts: Counter[Tuple[str, str]] = Counter()
    true_coverage = 0
    confs: List[float] = []
    conf_by_label: Dict[str, List[float]] = defaultdict(list)
    conf_by_label_canon: Dict[str, List[float]] = defaultdict(list)

    t0 = time.time()
    for p in files:
        try:
            data = p.read_bytes()
        except Exception as exc:
            rows.append(
                {
                    "file": p.name,
                    "status": "read_error",
                    "error": str(exc),
                }
            )
            status_counts["read_error"] += 1
            continue

        result = clf.predict_from_bytes(data, p.name)
        status = str(result.get("status") or "")
        status_counts[status] += 1

        pred_label = str(result.get("label") or "").strip()
        pred_label_canon = _canonical(pred_label, alias_map) if pred_label else ""
        pred_conf = _safe_float(result.get("confidence"), 0.0)
        temperature = _safe_float(result.get("temperature"), 1.0)
        temperature_source = str(result.get("temperature_source") or "")

        true_label = ""
        true_label_raw = ""
        true_label_conf = 0.0
        true_label_source = ""
        if bool(args.labels_from_parent_dir):
            true_label_raw = p.parent.name
            true_label_source = "parent_dir"
            true_label_conf = 1.0
        elif manifest_csv is not None and (manifest_labels_by_relpath or manifest_labels_by_name):
            rel = ""
            try:
                rel = str(p.relative_to(dxf_dir))
            except Exception:
                rel = ""
            row = manifest_labels_by_relpath.get(rel) or manifest_labels_by_name.get(p.name)
            if row:
                true_label_raw = str(row.get("label_cn") or "").strip()
                true_label_source = "manifest"
                true_label_conf = _safe_float(row.get("label_confidence"), 1.0) or 1.0
        elif bool(args.labels_from_filename) and filename_classifier is not None:
            payload = filename_classifier.predict(p.name)
            true_label_raw = str(payload.get("label") or "").strip()
            true_label_source = "filename"
            true_label_conf = _safe_float(payload.get("confidence"), 0.0)

        if true_label_raw and true_label_conf >= float(args.true_label_min_confidence):
            true_label = _canonical(true_label_raw, alias_map)
            true_counts[true_label] += 1
            true_coverage += 1

        if status == "ok" and pred_label:
            pred_counts[pred_label] += 1
            if pred_label_canon:
                pred_canon_counts[pred_label_canon] += 1
            confs.append(pred_conf)
            conf_by_label[pred_label].append(pred_conf)
            if pred_label_canon:
                conf_by_label_canon[pred_label_canon].append(pred_conf)
            if true_label:
                if pred_label_canon and pred_label_canon == true_label:
                    correct_counts[true_label] += 1
                elif pred_label_canon:
                    confusion_counts[(true_label, pred_label_canon)] += 1

        rows.append(
            {
                "file": p.name,
                "true_label": true_label,
                "true_label_raw": true_label_raw,
                "true_label_confidence": f"{true_label_conf:.4f}",
                "true_label_source": true_label_source,
                "pred_label": pred_label,
                "pred_label_canon": pred_label_canon,
                "pred_confidence": f"{pred_conf:.4f}",
                "status": status,
                "temperature": f"{temperature:.4f}",
                "temperature_source": temperature_source,
            }
        )

    elapsed_s = time.time() - t0
    ok = int(status_counts.get("ok", 0))
    acc = None
    if (
        bool(args.labels_from_parent_dir)
        or bool(args.labels_from_filename)
        or bool(manifest_csv)
    ) and true_counts:
        total_labeled = sum(true_counts.values())
        correct_total = sum(correct_counts.values())
        acc = float(correct_total) / float(total_labeled) if total_labeled else 0.0

    conf_sorted = sorted(confs)
    conf_p50 = conf_sorted[len(conf_sorted) // 2] if conf_sorted else 0.0
    conf_p90 = conf_sorted[int(len(conf_sorted) * 0.9)] if conf_sorted else 0.0

    top_confusions: List[Dict[str, Any]] = []
    for (t, pred), count in confusion_counts.most_common(25):
        top_confusions.append(
            {"true_label": t, "pred_label": pred, "count": int(count)}
        )

    truth_enabled = bool(args.labels_from_parent_dir) or bool(args.labels_from_filename) or bool(manifest_csv)

    summary: Dict[str, Any] = {
        "dxf_dir": (str(dxf_dir) if args.include_abs_paths else str(dxf_dir.name)),
        "model_path": str(args.model_path),
        "sampled_files": len(files),
        "elapsed_seconds": round(elapsed_s, 3),
        "status_counts": dict(status_counts),
        "label_map_size": len(getattr(clf, "label_map", {}) or {}),
        "top_pred_labels": _summarize_counts(pred_counts, topn=20),
        "top_pred_labels_canon": _summarize_counts(pred_canon_counts, topn=20),
        "confidence": {
            "count": len(confs),
            "p50": round(conf_p50, 4),
            "p90": round(conf_p90, 4),
        },
        "true_labels": (
            {
                "source": (
                    "parent_dir"
                    if args.labels_from_parent_dir
                    else (
                        "filename"
                        if args.labels_from_filename
                        else ("manifest" if manifest_csv else "")
                    )
                ),
                "min_confidence": float(args.true_label_min_confidence),
                "coverage": int(true_coverage),
                "coverage_rate": round(float(true_coverage) / float(len(files)), 6),
                "distinct_count": len(true_counts),
                "top_true_labels": _summarize_counts(true_counts, topn=20),
            }
            if truth_enabled
            else None
        ),
        "accuracy": acc if truth_enabled else None,
        "top_confusions": top_confusions if (truth_enabled and acc is not None) else None,
        "per_class_accuracy": {
            label: {
                "total": int(true_counts.get(label, 0)),
                "correct": int(correct_counts.get(label, 0)),
                "accuracy": round(
                    float(correct_counts.get(label, 0)) / float(true_counts.get(label, 1)),
                    4,
                ),
            }
            for label in sorted(true_counts.keys())
        }
        if (truth_enabled and acc is not None)
        else None,
    }

    _write_csv(out_dir / "predictions.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
