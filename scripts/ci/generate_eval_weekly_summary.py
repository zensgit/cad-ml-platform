#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


@dataclass
class WeeklyMetrics:
    total_reports: int
    ocr_reports: int
    combined_reports: int
    history_reports: int
    ocr_dimension_recall_mean: float
    ocr_brier_score_mean: float
    ocr_edge_f1_mean: float
    combined_score_mean: float
    combined_vision_score_mean: float
    combined_ocr_score_mean: float
    history_coverage_mean: float
    history_accuracy_mean: float
    history_macro_f1_mean: float
    hybrid_blind_reports: int
    hybrid_blind_accuracy_mean: float
    hybrid_blind_graph2d_accuracy_mean: float
    hybrid_blind_gain_mean: float
    hybrid_blind_coverage_mean: float
    hybrid_blind_label_slice_count_mean: float
    hybrid_blind_label_slice_count_latest: int
    hybrid_blind_family_slice_count_mean: float
    hybrid_blind_family_slice_count_latest: int


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def collect_metrics(eval_history_dir: Path, days: int, now: Optional[datetime] = None) -> WeeklyMetrics:
    ref_now = now or datetime.now(timezone.utc)
    cutoff = ref_now - timedelta(days=max(1, int(days)))

    total_reports = 0
    ocr_metrics: List[Dict[str, float]] = []
    combined_metrics: List[Dict[str, float]] = []
    history_metrics: List[Dict[str, float]] = []
    hybrid_blind_metrics: List[Dict[str, float]] = []
    latest_hybrid_blind_ts: Optional[datetime] = None
    latest_hybrid_blind_label_slice_count = 0
    latest_hybrid_blind_family_slice_count = 0

    for path in sorted(eval_history_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        ts = _parse_ts(payload.get("timestamp"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue

        total_reports += 1
        report_type = str(payload.get("type") or "").strip()
        if report_type == "ocr":
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            ocr_metrics.append(
                {
                    "dimension_recall": _safe_float(metrics.get("dimension_recall"), 0.0),
                    "brier_score": _safe_float(metrics.get("brier_score"), 0.0),
                    "edge_f1": _safe_float(metrics.get("edge_f1"), 0.0),
                }
            )
        elif report_type == "combined":
            combined = payload.get("combined") if isinstance(payload.get("combined"), dict) else {}
            combined_metrics.append(
                {
                    "combined_score": _safe_float(combined.get("combined_score"), 0.0),
                    "vision_score": _safe_float(combined.get("vision_score"), 0.0),
                    "ocr_score": _safe_float(combined.get("ocr_score"), 0.0),
                }
            )
        elif report_type == "history_sequence":
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            history_metrics.append(
                {
                    "coverage": _safe_float(metrics.get("coverage"), 0.0),
                    "accuracy_overall": _safe_float(metrics.get("accuracy_overall"), 0.0),
                    "macro_f1_overall": _safe_float(metrics.get("macro_f1_overall"), 0.0),
                }
            )
        elif report_type == "hybrid_blind":
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            label_slice_meta = (
                metrics.get("label_slice_meta")
                if isinstance(metrics.get("label_slice_meta"), dict)
                else {}
            )
            label_slice_count = 0
            if "slice_count" in label_slice_meta:
                label_slice_count = int(_safe_float(label_slice_meta.get("slice_count"), 0.0))
            else:
                raw_slices = metrics.get("label_slices")
                if isinstance(raw_slices, list):
                    label_slice_count = len(raw_slices)
            family_slice_meta = (
                metrics.get("family_slice_meta")
                if isinstance(metrics.get("family_slice_meta"), dict)
                else {}
            )
            family_slice_count = 0
            if "slice_count" in family_slice_meta:
                family_slice_count = int(
                    _safe_float(family_slice_meta.get("slice_count"), 0.0)
                )
            else:
                raw_family_slices = metrics.get("family_slices")
                if isinstance(raw_family_slices, list):
                    family_slice_count = len(raw_family_slices)
            hybrid_blind_metrics.append(
                {
                    "hybrid_accuracy": _safe_float(metrics.get("hybrid_accuracy"), 0.0),
                    "graph2d_accuracy": _safe_float(metrics.get("graph2d_accuracy"), 0.0),
                    "hybrid_gain_vs_graph2d": _safe_float(
                        metrics.get("hybrid_gain_vs_graph2d"), 0.0
                    ),
                    "weak_label_coverage": _safe_float(
                        metrics.get("weak_label_coverage"), 0.0
                    ),
                    "label_slice_count": float(max(0, label_slice_count)),
                    "family_slice_count": float(max(0, family_slice_count)),
                }
            )
            if latest_hybrid_blind_ts is None or ts > latest_hybrid_blind_ts:
                latest_hybrid_blind_ts = ts
                latest_hybrid_blind_label_slice_count = max(0, int(label_slice_count))
                latest_hybrid_blind_family_slice_count = max(0, int(family_slice_count))

    return WeeklyMetrics(
        total_reports=total_reports,
        ocr_reports=len(ocr_metrics),
        combined_reports=len(combined_metrics),
        history_reports=len(history_metrics),
        ocr_dimension_recall_mean=_mean([m["dimension_recall"] for m in ocr_metrics]),
        ocr_brier_score_mean=_mean([m["brier_score"] for m in ocr_metrics]),
        ocr_edge_f1_mean=_mean([m["edge_f1"] for m in ocr_metrics]),
        combined_score_mean=_mean([m["combined_score"] for m in combined_metrics]),
        combined_vision_score_mean=_mean([m["vision_score"] for m in combined_metrics]),
        combined_ocr_score_mean=_mean([m["ocr_score"] for m in combined_metrics]),
        history_coverage_mean=_mean([m["coverage"] for m in history_metrics]),
        history_accuracy_mean=_mean([m["accuracy_overall"] for m in history_metrics]),
        history_macro_f1_mean=_mean([m["macro_f1_overall"] for m in history_metrics]),
        hybrid_blind_reports=len(hybrid_blind_metrics),
        hybrid_blind_accuracy_mean=_mean([m["hybrid_accuracy"] for m in hybrid_blind_metrics]),
        hybrid_blind_graph2d_accuracy_mean=_mean(
            [m["graph2d_accuracy"] for m in hybrid_blind_metrics]
        ),
        hybrid_blind_gain_mean=_mean(
            [m["hybrid_gain_vs_graph2d"] for m in hybrid_blind_metrics]
        ),
        hybrid_blind_coverage_mean=_mean(
            [m["weak_label_coverage"] for m in hybrid_blind_metrics]
        ),
        hybrid_blind_label_slice_count_mean=_mean(
            [m["label_slice_count"] for m in hybrid_blind_metrics]
        ),
        hybrid_blind_label_slice_count_latest=latest_hybrid_blind_label_slice_count,
        hybrid_blind_family_slice_count_mean=_mean(
            [m["family_slice_count"] for m in hybrid_blind_metrics]
        ),
        hybrid_blind_family_slice_count_latest=latest_hybrid_blind_family_slice_count,
    )


def build_weekly_markdown(
    metrics: WeeklyMetrics,
    days: int,
    generated_at: str,
    context: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Weekly Evaluation Summary")
    lines.append("")
    lines.append(f"- Window: last `{days}` days")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Reports scanned: `{metrics.total_reports}`")
    lines.append("")
    lines.append("## Rolling Means")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| OCR reports | {metrics.ocr_reports} |")
    lines.append(f"| OCR dimension_recall mean | {metrics.ocr_dimension_recall_mean:.6f} |")
    lines.append(f"| OCR brier_score mean | {metrics.ocr_brier_score_mean:.6f} |")
    lines.append(f"| OCR edge_f1 mean | {metrics.ocr_edge_f1_mean:.6f} |")
    lines.append(f"| Combined reports | {metrics.combined_reports} |")
    lines.append(f"| Combined score mean | {metrics.combined_score_mean:.6f} |")
    lines.append(f"| Combined vision_score mean | {metrics.combined_vision_score_mean:.6f} |")
    lines.append(f"| Combined ocr_score mean | {metrics.combined_ocr_score_mean:.6f} |")
    lines.append(f"| History reports | {metrics.history_reports} |")
    lines.append(f"| History coverage mean | {metrics.history_coverage_mean:.6f} |")
    lines.append(f"| History accuracy mean | {metrics.history_accuracy_mean:.6f} |")
    lines.append(f"| History macro_f1 mean | {metrics.history_macro_f1_mean:.6f} |")
    lines.append(f"| Hybrid blind reports | {metrics.hybrid_blind_reports} |")
    lines.append(f"| Hybrid blind accuracy mean | {metrics.hybrid_blind_accuracy_mean:.6f} |")
    lines.append(
        f"| Hybrid blind graph2d accuracy mean | {metrics.hybrid_blind_graph2d_accuracy_mean:.6f} |"
    )
    lines.append(f"| Hybrid blind gain mean | {metrics.hybrid_blind_gain_mean:.6f} |")
    lines.append(
        f"| Hybrid blind weak-label coverage mean | {metrics.hybrid_blind_coverage_mean:.6f} |"
    )
    lines.append(
        f"| Hybrid blind label-slice count mean | {metrics.hybrid_blind_label_slice_count_mean:.6f} |"
    )
    lines.append(
        f"| Hybrid blind label-slice count latest | {metrics.hybrid_blind_label_slice_count_latest} |"
    )
    lines.append(
        f"| Hybrid blind family-slice count mean | {metrics.hybrid_blind_family_slice_count_mean:.6f} |"
    )
    lines.append(
        f"| Hybrid blind family-slice count latest | {metrics.hybrid_blind_family_slice_count_latest} |"
    )

    lines.append("")
    lines.append("## Current Gates Snapshot")
    lines.append("")
    lines.append("| Signal | Value |")
    lines.append("|---|---|")
    lines.append(f"| Graph2D blind gate | `{context.get('graph2d_blind_status', 'unknown')}` |")
    lines.append(f"| Graph2D blind accuracy | `{context.get('graph2d_blind_accuracy', 'n/a')}` |")
    lines.append(f"| Hybrid blind gate | `{context.get('hybrid_blind_status', 'unknown')}` |")
    lines.append(f"| Hybrid blind accuracy | `{context.get('hybrid_blind_accuracy', 'n/a')}` |")
    lines.append(f"| Hybrid blind gain vs Graph2D | `{context.get('hybrid_blind_gain', 'n/a')}` |")
    lines.append(
        f"| Hybrid calibration | `{context.get('hybrid_calibration_status', 'unknown')}` |"
    )
    lines.append(
        f"| Hybrid calibration gate | `{context.get('hybrid_calibration_gate_status', 'unknown')}` |"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate rolling weekly evaluation summary.")
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Evaluation history directory.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/weekly_summary.md",
        help="Output markdown path.",
    )
    parser.add_argument("--days", type=int, default=7, help="Rolling window in days.")
    parser.add_argument("--graph2d-blind-status", default="unknown")
    parser.add_argument("--graph2d-blind-accuracy", default="n/a")
    parser.add_argument("--hybrid-blind-status", default="unknown")
    parser.add_argument("--hybrid-blind-accuracy", default="n/a")
    parser.add_argument("--hybrid-blind-gain", default="n/a")
    parser.add_argument("--hybrid-calibration-status", default="unknown")
    parser.add_argument("--hybrid-calibration-gate-status", default="unknown")
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_history_dir)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    metrics = collect_metrics(eval_dir, days=max(1, int(args.days)))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    context = {
        "graph2d_blind_status": args.graph2d_blind_status,
        "graph2d_blind_accuracy": args.graph2d_blind_accuracy,
        "hybrid_blind_status": args.hybrid_blind_status,
        "hybrid_blind_accuracy": args.hybrid_blind_accuracy,
        "hybrid_blind_gain": args.hybrid_blind_gain,
        "hybrid_calibration_status": args.hybrid_calibration_status,
        "hybrid_calibration_gate_status": args.hybrid_calibration_gate_status,
    }
    text = build_weekly_markdown(
        metrics=metrics,
        days=max(1, int(args.days)),
        generated_at=generated_at,
        context=context,
    )
    out_md.write_text(text, encoding="utf-8")
    print(f"output={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
