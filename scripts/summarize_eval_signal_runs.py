#!/usr/bin/env python3
"""Summarize combined/OCR/hybrid-blind eval-history records into a canonical report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from scripts.eval_report_data_helpers import load_json_dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _normalize_hybrid_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    label_slice_meta = (
        metrics.get("label_slice_meta") if isinstance(metrics.get("label_slice_meta"), dict) else {}
    )
    family_slice_meta = (
        metrics.get("family_slice_meta") if isinstance(metrics.get("family_slice_meta"), dict) else {}
    )

    label_slice_count = 0
    if "slice_count" in label_slice_meta:
        label_slice_count = int(_safe_float(label_slice_meta.get("slice_count"), 0.0))
    else:
        raw_slices = metrics.get("label_slices")
        if isinstance(raw_slices, list):
            label_slice_count = len(raw_slices)

    family_slice_count = 0
    if "slice_count" in family_slice_meta:
        family_slice_count = int(_safe_float(family_slice_meta.get("slice_count"), 0.0))
    else:
        raw_family_slices = metrics.get("family_slices")
        if isinstance(raw_family_slices, list):
            family_slice_count = len(raw_family_slices)

    return {
        "hybrid_accuracy": _safe_float(metrics.get("hybrid_accuracy"), 0.0),
        "graph2d_accuracy": _safe_float(metrics.get("graph2d_accuracy"), 0.0),
        "hybrid_gain_vs_graph2d": _safe_float(metrics.get("hybrid_gain_vs_graph2d"), 0.0),
        "weak_label_coverage": _safe_float(metrics.get("weak_label_coverage"), 0.0),
        "label_slice_count": max(0, int(label_slice_count)),
        "family_slice_count": max(0, int(family_slice_count)),
    }


def _normalize_report_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_path": str(row.get("report_path") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "report_type": str(row.get("report_type") or ""),
        "branch": str(row.get("branch") or ""),
        "commit": str(row.get("commit") or ""),
        "combined": dict(row.get("combined") or {}),
        "metrics": dict(row.get("metrics") or {}),
        "run_context": dict(row.get("run_context") or {}),
    }


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    payload = load_json_dict(path)
    if not payload:
        return None

    report_type = str(payload.get("type") or "").strip()
    if (
        report_type == "combined"
        or "combined" in payload
        or path.name.endswith("_combined.json")
    ):
        combined = payload.get("combined") if isinstance(payload.get("combined"), dict) else {}
        return {
            "report_path": str(path),
            "timestamp": str(payload.get("timestamp") or ""),
            "report_type": "combined",
            "branch": str(payload.get("branch") or ""),
            "commit": str(payload.get("commit") or ""),
            "combined": {
                "combined_score": _safe_float(combined.get("combined_score"), 0.0),
                "vision_score": _safe_float(combined.get("vision_score"), 0.0),
                "ocr_score": _safe_float(combined.get("ocr_score"), 0.0),
                "vision_weight": _safe_float(combined.get("vision_weight"), 0.5),
                "ocr_weight": _safe_float(combined.get("ocr_weight"), 0.5),
            },
            "metrics": {},
            "run_context": dict(payload.get("run_context") or {}),
        }

    is_legacy_ocr = report_type == "" and "metrics" in payload and "history_metrics" not in payload
    if report_type == "ocr" or is_legacy_ocr:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        return {
            "report_path": str(path),
            "timestamp": str(payload.get("timestamp") or ""),
            "report_type": "ocr",
            "branch": str(payload.get("branch") or ""),
            "commit": str(payload.get("commit") or ""),
            "combined": {},
            "metrics": {
                "dimension_recall": _safe_float(metrics.get("dimension_recall"), 0.0),
                "brier_score": _safe_float(metrics.get("brier_score"), 0.0),
                "edge_f1": _safe_float(metrics.get("edge_f1"), 0.0),
            },
            "run_context": dict(payload.get("run_context") or {}),
        }

    if report_type == "hybrid_blind":
        return {
            "report_path": str(path),
            "timestamp": str(payload.get("timestamp") or ""),
            "report_type": "hybrid_blind",
            "branch": str(payload.get("branch") or ""),
            "commit": str(payload.get("commit") or ""),
            "combined": {},
            "metrics": _normalize_hybrid_metrics(payload),
            "run_context": dict(payload.get("run_context") or {}),
        }

    return None


def _collect_reports(eval_history_dir: Path, report_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not eval_history_dir.exists():
        return rows
    for path in sorted(eval_history_dir.glob(str(report_glob))):
        report = _load_report(path)
        if report is not None:
            rows.append(report)
    return rows


def _build_report_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _normalize_report_row(row)
        for row in sorted(rows, key=lambda candidate: _parse_ts(candidate.get("timestamp")))
    ]


def _rows_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rows = summary.get("report_rows")
    if not isinstance(raw_rows, list):
        return []
    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(_normalize_report_row(row))
    return rows


def _load_summary_artifact(path: Path) -> Optional[Dict[str, Any]]:
    payload = load_json_dict(path)
    if not payload:
        return None
    if str(payload.get("status") or "").strip() != "ok":
        return None
    if str(payload.get("surface_kind") or "").strip() != "eval_signal_experiment_summary":
        return None
    return payload


def _load_or_build_summary(
    summary_json_path: Path, *, eval_history_dir: Path, report_glob: str
) -> Dict[str, Any]:
    loaded = _load_summary_artifact(summary_json_path)
    if loaded is not None:
        return loaded
    rows = _collect_reports(eval_history_dir, report_glob)
    return _build_summary(rows, eval_history_dir=eval_history_dir, report_glob=report_glob)


def _mean_from_rows(rows: Sequence[Dict[str, Any]], report_type: str, field: str, *, source: str) -> float:
    values: List[float] = []
    for row in rows:
        if str(row.get("report_type") or "") != report_type:
            continue
        container = row.get(source) if isinstance(row.get(source), dict) else {}
        values.append(_safe_float(container.get(field), 0.0))
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _build_window_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    combined_rows = [row for row in rows_list if str(row.get("report_type") or "") == "combined"]
    ocr_rows = [row for row in rows_list if str(row.get("report_type") or "") == "ocr"]
    hybrid_rows = [row for row in rows_list if str(row.get("report_type") or "") == "hybrid_blind"]
    latest_hybrid = (
        max(hybrid_rows, key=lambda row: _parse_ts(row.get("timestamp"))) if hybrid_rows else {}
    )
    latest_hybrid_metrics = (
        latest_hybrid.get("metrics") if isinstance(latest_hybrid.get("metrics"), dict) else {}
    )
    return {
        "report_count": int(len(rows_list)),
        "combined_reports": int(len(combined_rows)),
        "ocr_reports": int(len(ocr_rows)),
        "hybrid_blind_reports": int(len(hybrid_rows)),
        "combined_score_mean": _mean_from_rows(rows_list, "combined", "combined_score", source="combined"),
        "combined_vision_score_mean": _mean_from_rows(
            rows_list, "combined", "vision_score", source="combined"
        ),
        "combined_ocr_score_mean": _mean_from_rows(rows_list, "combined", "ocr_score", source="combined"),
        "ocr_dimension_recall_mean": _mean_from_rows(
            rows_list, "ocr", "dimension_recall", source="metrics"
        ),
        "ocr_brier_score_mean": _mean_from_rows(rows_list, "ocr", "brier_score", source="metrics"),
        "ocr_edge_f1_mean": _mean_from_rows(rows_list, "ocr", "edge_f1", source="metrics"),
        "hybrid_blind_accuracy_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "hybrid_accuracy", source="metrics"
        ),
        "hybrid_blind_graph2d_accuracy_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "graph2d_accuracy", source="metrics"
        ),
        "hybrid_blind_gain_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "hybrid_gain_vs_graph2d", source="metrics"
        ),
        "hybrid_blind_coverage_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "weak_label_coverage", source="metrics"
        ),
        "hybrid_blind_label_slice_count_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "label_slice_count", source="metrics"
        ),
        "hybrid_blind_label_slice_count_latest": int(
            _safe_float(latest_hybrid_metrics.get("label_slice_count"), 0.0)
        ),
        "hybrid_blind_family_slice_count_mean": _mean_from_rows(
            rows_list, "hybrid_blind", "family_slice_count", source="metrics"
        ),
        "hybrid_blind_family_slice_count_latest": int(
            _safe_float(latest_hybrid_metrics.get("family_slice_count"), 0.0)
        ),
    }


def _build_summary(
    rows: List[Dict[str, Any]], *, eval_history_dir: Path, report_glob: str
) -> Dict[str, Any]:
    window = _build_window_summary(rows)
    latest_combined = (
        max(
            [row for row in rows if str(row.get("report_type") or "") == "combined"],
            key=lambda row: _parse_ts(row.get("timestamp")),
        )
        if any(str(row.get("report_type") or "") == "combined" for row in rows)
        else None
    )
    latest_ocr = (
        max(
            [row for row in rows if str(row.get("report_type") or "") == "ocr"],
            key=lambda row: _parse_ts(row.get("timestamp")),
        )
        if any(str(row.get("report_type") or "") == "ocr" for row in rows)
        else None
    )
    latest_hybrid = (
        max(
            [row for row in rows if str(row.get("report_type") or "") == "hybrid_blind"],
            key=lambda row: _parse_ts(row.get("timestamp")),
        )
        if any(str(row.get("report_type") or "") == "hybrid_blind" for row in rows)
        else None
    )
    return {
        "status": "ok",
        "surface_kind": "eval_signal_experiment_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "report_glob": report_glob,
        "report_count": int(window["report_count"]),
        "report_counts": {
            "combined": int(window["combined_reports"]),
            "ocr": int(window["ocr_reports"]),
            "hybrid_blind": int(window["hybrid_blind_reports"]),
        },
        "aggregate_metrics": {
            "combined_score_mean": _safe_float(window["combined_score_mean"], 0.0),
            "combined_vision_score_mean": _safe_float(window["combined_vision_score_mean"], 0.0),
            "combined_ocr_score_mean": _safe_float(window["combined_ocr_score_mean"], 0.0),
            "ocr_dimension_recall_mean": _safe_float(window["ocr_dimension_recall_mean"], 0.0),
            "ocr_brier_score_mean": _safe_float(window["ocr_brier_score_mean"], 0.0),
            "ocr_edge_f1_mean": _safe_float(window["ocr_edge_f1_mean"], 0.0),
            "hybrid_blind_accuracy_mean": _safe_float(window["hybrid_blind_accuracy_mean"], 0.0),
            "hybrid_blind_graph2d_accuracy_mean": _safe_float(
                window["hybrid_blind_graph2d_accuracy_mean"], 0.0
            ),
            "hybrid_blind_gain_mean": _safe_float(window["hybrid_blind_gain_mean"], 0.0),
            "hybrid_blind_coverage_mean": _safe_float(window["hybrid_blind_coverage_mean"], 0.0),
            "hybrid_blind_label_slice_count_mean": _safe_float(
                window["hybrid_blind_label_slice_count_mean"], 0.0
            ),
            "hybrid_blind_family_slice_count_mean": _safe_float(
                window["hybrid_blind_family_slice_count_mean"], 0.0
            ),
        },
        "report_rows": _build_report_rows(rows),
        "latest_combined_run": dict(latest_combined) if isinstance(latest_combined, dict) else None,
        "latest_ocr_run": dict(latest_ocr) if isinstance(latest_ocr, dict) else None,
        "latest_hybrid_blind_run": dict(latest_hybrid) if isinstance(latest_hybrid, dict) else None,
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    aggregate = summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    report_counts = summary.get("report_counts") if isinstance(summary.get("report_counts"), dict) else {}
    latest_combined = (
        summary.get("latest_combined_run") if isinstance(summary.get("latest_combined_run"), dict) else {}
    )
    latest_ocr = summary.get("latest_ocr_run") if isinstance(summary.get("latest_ocr_run"), dict) else {}
    latest_hybrid = (
        summary.get("latest_hybrid_blind_run")
        if isinstance(summary.get("latest_hybrid_blind_run"), dict)
        else {}
    )
    lines = [
        "# Eval Signal Experiment Summary",
        "",
        f"- Eval history dir: `{summary.get('eval_history_dir', '')}`",
        f"- Report glob: `{summary.get('report_glob', '')}`",
        f"- Report count: `{summary.get('report_count', 0)}`",
        f"- Combined reports: `{report_counts.get('combined', 0)}`",
        f"- OCR reports: `{report_counts.get('ocr', 0)}`",
        f"- Hybrid blind reports: `{report_counts.get('hybrid_blind', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Combined score mean | {_safe_float(aggregate.get('combined_score_mean'), 0.0):.6f} |",
        f"| Combined vision score mean | {_safe_float(aggregate.get('combined_vision_score_mean'), 0.0):.6f} |",
        f"| Combined OCR score mean | {_safe_float(aggregate.get('combined_ocr_score_mean'), 0.0):.6f} |",
        f"| OCR dimension recall mean | {_safe_float(aggregate.get('ocr_dimension_recall_mean'), 0.0):.6f} |",
        f"| OCR brier score mean | {_safe_float(aggregate.get('ocr_brier_score_mean'), 0.0):.6f} |",
        f"| OCR edge F1 mean | {_safe_float(aggregate.get('ocr_edge_f1_mean'), 0.0):.6f} |",
        f"| Hybrid blind accuracy mean | {_safe_float(aggregate.get('hybrid_blind_accuracy_mean'), 0.0):.6f} |",
        f"| Hybrid blind graph2d accuracy mean | {_safe_float(aggregate.get('hybrid_blind_graph2d_accuracy_mean'), 0.0):.6f} |",
        f"| Hybrid blind gain mean | {_safe_float(aggregate.get('hybrid_blind_gain_mean'), 0.0):.6f} |",
        "",
        "## Latest Runs",
        "",
        f"- Latest combined run: `{latest_combined.get('timestamp', '')}`",
        f"- Latest OCR run: `{latest_ocr.get('timestamp', '')}`",
        f"- Latest hybrid blind run: `{latest_hybrid.get('timestamp', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize eval signal records.")
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history JSON records.",
    )
    parser.add_argument(
        "--report-glob",
        default="*.json",
        help="Glob pattern used under --eval-history-dir.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/eval_history/eval_signal_experiment_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/eval_signal_experiment_summary.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    rows = _collect_reports(eval_history_dir, str(args.report_glob))
    summary = _build_summary(rows, eval_history_dir=eval_history_dir, report_glob=str(args.report_glob))
    output_json = Path(str(args.output_json))
    output_md = Path(str(args.output_md))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"Eval signal summary JSON: {output_json}")
    print(f"Eval signal summary Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
