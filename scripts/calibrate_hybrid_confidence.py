#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.hybrid.calibration import CalibrationMethod, ConfidenceCalibrator


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if all(ord(ch) < 128 for ch in text):
        return text.lower()
    return text


def _parse_boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "t", "ok", "correct", "match"}:
        return True
    if text in {"0", "false", "no", "n", "f", "wrong", "mismatch"}:
        return False
    return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _count_non_empty(rows: List[Dict[str, Any]], column: str) -> int:
    count = 0
    for row in rows:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            count += 1
    return count


def _resolve_column(
    rows: List[Dict[str, Any]],
    preferred: str,
    fallbacks: List[str],
) -> str:
    if rows:
        preferred_count = _count_non_empty(rows, preferred)
        if preferred_count > 0:
            return preferred
    for candidate in fallbacks:
        if not candidate:
            continue
        if rows and _count_non_empty(rows, candidate) > 0:
            return candidate
    return preferred


def _load_rows(path: Path, max_rows: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows > 0 and idx >= max_rows:
                break
            rows.append(dict(row))
    return rows


def _extract_sample(
    row: Dict[str, Any],
    *,
    confidence_col: str,
    correct_col: str,
    pred_label_col: str,
    truth_label_col: str,
    source_col: str,
) -> Optional[Tuple[float, int, str, str]]:
    conf_raw = row.get(confidence_col)
    if conf_raw is None or str(conf_raw).strip() == "":
        return None
    confidence = _safe_float(conf_raw, default=-1.0)
    if confidence < 0:
        return None
    confidence = min(1.0, max(0.0, confidence))

    source = str(row.get(source_col) or "").strip() or "unknown"

    correct: Optional[bool] = None
    if correct_col:
        correct = _parse_boolish(row.get(correct_col))

    pred = _normalize_label(row.get(pred_label_col))
    truth = _normalize_label(row.get(truth_label_col))
    if correct is None and pred and truth:
        correct = pred == truth

    if correct is None:
        return None

    return confidence, int(bool(correct)), source, ("match" if correct else "mismatch")


def calibrate(
    rows: List[Dict[str, Any]],
    *,
    method: CalibrationMethod,
    per_source: bool,
    confidence_col: str,
    correct_col: str,
    pred_label_col: str,
    truth_label_col: str,
    source_col: str,
    min_samples: int,
    min_samples_per_source: int,
) -> Dict[str, Any]:
    parsed: List[Tuple[float, int, str, str]] = []
    dropped_no_label = 0
    dropped_bad_conf = 0

    for row in rows:
        sample = _extract_sample(
            row,
            confidence_col=confidence_col,
            correct_col=correct_col,
            pred_label_col=pred_label_col,
            truth_label_col=truth_label_col,
            source_col=source_col,
        )
        if sample is None:
            conf_raw = row.get(confidence_col)
            if conf_raw is None or str(conf_raw).strip() == "":
                dropped_bad_conf += 1
            else:
                dropped_no_label += 1
            continue
        parsed.append(sample)

    confidences = np.asarray([item[0] for item in parsed], dtype=float)
    labels = np.asarray([item[1] for item in parsed], dtype=int)
    sources = np.asarray([item[2] for item in parsed], dtype=object)
    pair_counts: Dict[str, int] = {}
    for _, _, _, tag in parsed:
        pair_counts[tag] = pair_counts.get(tag, 0) + 1

    if len(parsed) < min_samples:
        return {
            "status": "insufficient_samples",
            "n_samples": int(len(parsed)),
            "min_samples": int(min_samples),
            "dropped_bad_confidence": int(dropped_bad_conf),
            "dropped_no_correctness": int(dropped_no_label),
            "pair_counts": pair_counts,
            "source_counts": {},
        }

    source_counts: Dict[str, int] = {}
    for src in sources:
        source_counts[str(src)] = source_counts.get(str(src), 0) + 1

    effective_per_source = bool(per_source)
    if per_source:
        eligible_sources = [
            src
            for src, count in source_counts.items()
            if count >= min_samples_per_source
        ]
        if len(eligible_sources) == 0:
            effective_per_source = False

    calibrator = ConfidenceCalibrator(method=method, per_source=effective_per_source)
    calibrator.fit(
        confidences,
        labels,
        sources=sources if effective_per_source else None,
    )

    metrics_before = calibrator.evaluate(confidences, labels).to_dict()
    calibrated_scores = np.asarray(
        [
            calibrator.calibrate(float(conf), source=str(src))
            for conf, src in zip(confidences, sources)
        ],
        dtype=float,
    )
    metrics_after = calibrator.evaluate(calibrated_scores, labels).to_dict()

    source_temperatures: Dict[str, Dict[str, Any]] = {}
    global_temperature: Optional[float] = None

    try:
        global_cal = getattr(calibrator, "_global_calibrator", None)
        if global_cal is not None and hasattr(global_cal, "temperature"):
            global_temperature = float(getattr(global_cal, "temperature"))
    except Exception:
        global_temperature = None

    for src, cal in getattr(calibrator, "_source_calibrators", {}).items():
        if hasattr(cal, "temperature"):
            try:
                source_temperatures[str(src)] = {
                    "temperature": float(getattr(cal, "temperature")),
                    "n_samples": int(source_counts.get(str(src), 0)),
                }
            except Exception:
                continue

    return {
        "status": "ok",
        "n_samples": int(len(parsed)),
        "dropped_bad_confidence": int(dropped_bad_conf),
        "dropped_no_correctness": int(dropped_no_label),
        "pair_counts": pair_counts,
        "source_counts": source_counts,
        "effective_per_source": bool(effective_per_source),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "temperature": global_temperature,
        "source_temperatures": source_temperatures,
        "fit_confidences": [round(float(v), 8) for v in confidences.tolist()],
        "fit_labels": [int(v) for v in labels.tolist()],
        "fit_sources": [str(v) for v in sources.tolist()],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit and export HybridClassifier confidence calibration."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV for calibration.")
    parser.add_argument(
        "--output-json",
        default="models/calibration/hybrid_confidence_calibration.json",
        help="Output calibration JSON path.",
    )
    parser.add_argument(
        "--method",
        default="temperature_scaling",
        choices=[m.value for m in CalibrationMethod],
        help="Calibration method.",
    )
    parser.add_argument(
        "--per-source",
        action="store_true",
        default=False,
        help="Fit per-source calibrators when enough samples exist.",
    )
    parser.add_argument(
        "--confidence-col", default="confidence", help="Confidence column name."
    )
    parser.add_argument(
        "--correct-col",
        default="is_correct",
        help="Correctness boolean column name (fallback to label comparison).",
    )
    parser.add_argument(
        "--pred-label-col",
        default="predicted_label",
        help="Predicted label column name.",
    )
    parser.add_argument(
        "--truth-label-col", default="correct_label", help="Ground truth label column."
    )
    parser.add_argument("--source-col", default="source", help="Source column name.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum usable samples for calibration.",
    )
    parser.add_argument(
        "--min-samples-per-source",
        type=int,
        default=10,
        help="Minimum samples per source for per-source calibration.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit rows loaded from CSV (0 means no limit).",
    )
    parser.add_argument(
        "--fail-on-insufficient-data",
        action="store_true",
        help="Exit non-zero when samples are insufficient.",
    )
    parser.add_argument(
        "--include-fit-data",
        action="store_true",
        help="Include fit arrays in output JSON (for fallback runtime fitting).",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_csv).expanduser()
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_path, max_rows=max(0, int(args.max_rows)))
    resolved_confidence_col = _resolve_column(
        rows,
        str(args.confidence_col),
        ["graph2d_confidence", "soft_override_confidence"],
    )
    resolved_correct_col = _resolve_column(
        rows,
        str(args.correct_col),
        ["agree_with_graph2d", "is_match", "match", "agree"],
    )
    resolved_pred_col = _resolve_column(
        rows,
        str(args.pred_label_col),
        ["graph2d_label", "soft_override_label", "part_type"],
    )
    resolved_truth_col = _resolve_column(
        rows,
        str(args.truth_label_col),
        ["ground_truth_label", "label", "part_type"],
    )
    resolved_source_col = _resolve_column(
        rows,
        str(args.source_col),
        ["primary_source", "decision_source"],
    )
    result = calibrate(
        rows,
        method=CalibrationMethod(args.method),
        per_source=bool(args.per_source),
        confidence_col=resolved_confidence_col,
        correct_col=resolved_correct_col,
        pred_label_col=resolved_pred_col,
        truth_label_col=resolved_truth_col,
        source_col=resolved_source_col,
        min_samples=max(1, int(args.min_samples)),
        min_samples_per_source=max(1, int(args.min_samples_per_source)),
    )

    payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": result.get("status"),
        "method": args.method,
        "per_source": bool(args.per_source),
        "input_csv": str(input_path),
        "n_rows": int(len(rows)),
        "n_samples": int(result.get("n_samples", 0) or 0),
        "min_samples": int(args.min_samples),
        "min_samples_per_source": int(args.min_samples_per_source),
        "effective_per_source": bool(result.get("effective_per_source", False)),
        "dropped_bad_confidence": int(result.get("dropped_bad_confidence", 0) or 0),
        "dropped_no_correctness": int(result.get("dropped_no_correctness", 0) or 0),
        "pair_counts": dict(result.get("pair_counts") or {}),
        "source_counts": dict(result.get("source_counts") or {}),
        "metrics_before": dict(result.get("metrics_before") or {}),
        "metrics_after": dict(result.get("metrics_after") or {}),
        "temperature": result.get("temperature"),
        "source_temperatures": dict(result.get("source_temperatures") or {}),
        "resolved_columns": {
            "confidence_col": resolved_confidence_col,
            "correct_col": resolved_correct_col,
            "pred_label_col": resolved_pred_col,
            "truth_label_col": resolved_truth_col,
            "source_col": resolved_source_col,
        },
    }
    if args.include_fit_data:
        payload["fit_confidences"] = list(result.get("fit_confidences") or [])
        payload["fit_labels"] = list(result.get("fit_labels") or [])
        payload["fit_sources"] = list(result.get("fit_sources") or [])

    output_path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"status={payload['status']}")
    print(f"n_rows={payload['n_rows']}")
    print(f"n_samples={payload['n_samples']}")
    print(f"output={output_path}")
    if payload["status"] == "ok":
        after = payload.get("metrics_after") or {}
        print(
            "metrics_after: ece={:.6f}, brier={:.6f}, mce={:.6f}".format(
                _safe_float(after.get("ece"), 0.0),
                _safe_float(after.get("brier_score"), 0.0),
                _safe_float(after.get("mce"), 0.0),
            )
        )
        return 0

    if args.fail_on_insufficient_data:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
