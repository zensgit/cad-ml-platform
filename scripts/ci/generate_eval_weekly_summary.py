#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.history_sequence_reporting_helpers import load_history_sequence_reporting_assets
from scripts import summarize_eval_signal_runs as eval_signal_canonical
from scripts import summarize_history_sequence_runs as history_canonical


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
    history_named_explainability_rate_mean: float
    history_named_error_rate_mean: float
    history_named_low_conf_rate_mean: float
    history_sequence_surface_kind_latest: str
    history_named_vocabulary_kind_latest: str
    history_named_worst_primary_family_latest: str
    history_named_worst_primary_reference_surface_latest: str
    history_named_worst_primary_status_latest: str
    history_surface_group_count: int
    history_best_surface_key_by_mean_accuracy_overall: str
    history_surface_groups: List[Dict[str, Any]]
    hybrid_blind_reports: int
    hybrid_blind_accuracy_mean: float
    hybrid_blind_graph2d_accuracy_mean: float
    hybrid_blind_gain_mean: float
    hybrid_blind_coverage_mean: float
    hybrid_blind_label_slice_count_mean: float
    hybrid_blind_label_slice_count_latest: int
    hybrid_blind_family_slice_count_mean: float
    hybrid_blind_family_slice_count_latest: int
def collect_metrics(
    eval_history_dir: Path,
    days: int,
    now: Optional[datetime] = None,
    eval_signal_summary_json: Optional[Path] = None,
    eval_signal_summary: Optional[Dict[str, Any]] = None,
    history_sequence_summary_json: Optional[Path] = None,
    history_sequence_reporting_bundle_json: Optional[Path] = None,
    history_sequence_summary: Optional[Dict[str, Any]] = None,
) -> WeeklyMetrics:
    ref_now = now or datetime.now(timezone.utc)
    cutoff = ref_now - timedelta(days=max(1, int(days)))
    if isinstance(eval_signal_summary, dict):
        signal_summary = eval_signal_summary
    else:
        signal_summary = eval_signal_canonical._load_or_build_summary(
            eval_signal_summary_json or (eval_history_dir / "eval_signal_experiment_summary.json"),
            eval_history_dir=eval_history_dir,
            report_glob="*.json",
        )
    signal_rows = [
        row
        for row in eval_signal_canonical._rows_from_summary(signal_summary)
        if (_parse_ts(row.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc)) >= cutoff
    ]
    signal_window = eval_signal_canonical._build_window_summary(signal_rows)

    if isinstance(history_sequence_summary, dict):
        history_summary = history_sequence_summary
    else:
        if history_sequence_summary_json is not None:
            history_summary = history_canonical._load_or_build_summary(
                history_sequence_summary_json,
                eval_history_dir=eval_history_dir,
                report_glob="*.json",
            )
        else:
            bundle, bundled_summary, _ = load_history_sequence_reporting_assets(
                eval_history_dir,
                bundle_json_path=history_sequence_reporting_bundle_json,
            )
            if bundle and bundled_summary:
                history_summary = bundled_summary
            else:
                history_summary = history_canonical._load_or_build_summary(
                    eval_history_dir / "history_sequence_experiment_summary.json",
                    eval_history_dir=eval_history_dir,
                    report_glob="*.json",
                )
    history_rows = [
        row
        for row in history_canonical._rows_from_summary(history_summary)
        if (_parse_ts(row.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc)) >= cutoff
    ]
    history_window = history_canonical._build_window_summary(history_rows)
    total_reports = int(signal_window["report_count"]) + int(history_window["report_count"])

    return WeeklyMetrics(
        total_reports=total_reports,
        ocr_reports=int(signal_window["ocr_reports"]),
        combined_reports=int(signal_window["combined_reports"]),
        history_reports=int(history_window["report_count"]),
        ocr_dimension_recall_mean=_safe_float(signal_window["ocr_dimension_recall_mean"], 0.0),
        ocr_brier_score_mean=_safe_float(signal_window["ocr_brier_score_mean"], 0.0),
        ocr_edge_f1_mean=_safe_float(signal_window["ocr_edge_f1_mean"], 0.0),
        combined_score_mean=_safe_float(signal_window["combined_score_mean"], 0.0),
        combined_vision_score_mean=_safe_float(signal_window["combined_vision_score_mean"], 0.0),
        combined_ocr_score_mean=_safe_float(signal_window["combined_ocr_score_mean"], 0.0),
        history_coverage_mean=_safe_float(history_window["coverage_mean"], 0.0),
        history_accuracy_mean=_safe_float(history_window["accuracy_mean"], 0.0),
        history_macro_f1_mean=_safe_float(history_window["macro_f1_mean"], 0.0),
        history_named_explainability_rate_mean=_safe_float(
            history_window["named_explainability_rate_mean"], 0.0
        ),
        history_named_error_rate_mean=_safe_float(history_window["named_error_rate_mean"], 0.0),
        history_named_low_conf_rate_mean=_safe_float(
            history_window["named_low_conf_rate_mean"], 0.0
        ),
        history_sequence_surface_kind_latest=str(history_window["latest_sequence_surface_kind"]),
        history_named_vocabulary_kind_latest=str(history_window["latest_named_vocabulary_kind"]),
        history_named_worst_primary_family_latest=str(
            history_window["latest_worst_primary_family"]
        ),
        history_named_worst_primary_reference_surface_latest=str(
            history_window["latest_worst_primary_reference_surface"]
        ),
        history_named_worst_primary_status_latest=str(
            history_window["latest_worst_primary_status"]
        ),
        history_surface_group_count=int(history_window["surface_group_count"]),
        history_best_surface_key_by_mean_accuracy_overall=str(
            history_window["best_surface_key_by_mean_accuracy_overall"]
        ),
        history_surface_groups=list(history_window["surface_groups"]),
        hybrid_blind_reports=int(signal_window["hybrid_blind_reports"]),
        hybrid_blind_accuracy_mean=_safe_float(signal_window["hybrid_blind_accuracy_mean"], 0.0),
        hybrid_blind_graph2d_accuracy_mean=_safe_float(
            signal_window["hybrid_blind_graph2d_accuracy_mean"], 0.0
        ),
        hybrid_blind_gain_mean=_safe_float(signal_window["hybrid_blind_gain_mean"], 0.0),
        hybrid_blind_coverage_mean=_safe_float(signal_window["hybrid_blind_coverage_mean"], 0.0),
        hybrid_blind_label_slice_count_mean=_safe_float(
            signal_window["hybrid_blind_label_slice_count_mean"], 0.0
        ),
        hybrid_blind_label_slice_count_latest=int(
            signal_window["hybrid_blind_label_slice_count_latest"]
        ),
        hybrid_blind_family_slice_count_mean=_safe_float(
            signal_window["hybrid_blind_family_slice_count_mean"], 0.0
        ),
        hybrid_blind_family_slice_count_latest=int(
            signal_window["hybrid_blind_family_slice_count_latest"]
        ),
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
    lines.append(
        f"| History named explainability rate mean | {metrics.history_named_explainability_rate_mean:.6f} |"
    )
    lines.append(
        f"| History named error rate mean | {metrics.history_named_error_rate_mean:.6f} |"
    )
    lines.append(
        f"| History named low-conf rate mean | {metrics.history_named_low_conf_rate_mean:.6f} |"
    )
    lines.append(
        f"| History sequence surface latest | `{metrics.history_sequence_surface_kind_latest or 'n/a'}` |"
    )
    lines.append(
        f"| History named vocabulary latest | `{metrics.history_named_vocabulary_kind_latest or 'n/a'}` |"
    )
    lines.append(
        f"| History worst family latest | `{metrics.history_named_worst_primary_family_latest or 'n/a'}` |"
    )
    lines.append(
        f"| History worst reference surface latest | `{metrics.history_named_worst_primary_reference_surface_latest or 'n/a'}` |"
    )
    lines.append(
        f"| History worst status latest | `{metrics.history_named_worst_primary_status_latest or 'n/a'}` |"
    )
    lines.append(
        f"| History surface group count | {metrics.history_surface_group_count} |"
    )
    lines.append(
        f"| History best surface key | `{metrics.history_best_surface_key_by_mean_accuracy_overall or 'n/a'}` |"
    )
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
    lines.append("## History Surface Groups")
    lines.append("")
    for group in metrics.history_surface_groups:
        lines.append(
            f"- `{group.get('surface_key', '')}`: reports=`{group.get('report_count', 0)}`, "
            f"mean_accuracy=`{_safe_float(group.get('mean_accuracy_overall'), 0.0):.6f}`, "
            f"mean_named_explainability=`{_safe_float(group.get('mean_named_explainability_rate'), 0.0):.6f}`, "
            f"latest=`{group.get('latest_timestamp', '')}`"
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
    parser.add_argument(
        "--eval-signal-summary-json",
        default="",
        help=(
            "Optional canonical eval-signal experiment summary JSON. "
            "When present, weekly combined/OCR/hybrid metrics prefer the canonical artifact."
        ),
    )
    parser.add_argument(
        "--history-sequence-summary-json",
        default="",
        help=(
            "Optional canonical history-sequence experiment summary JSON. "
            "When present, weekly history-sequence metrics prefer the canonical artifact."
        ),
    )
    parser.add_argument(
        "--history-sequence-reporting-bundle-json",
        default="",
        help=(
            "Optional history-sequence reporting bundle JSON. When summary JSON is not "
            "explicitly set, weekly history-sequence metrics prefer the bundle-referenced "
            "canonical summary artifact."
        ),
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

    eval_signal_summary_json = (
        Path(str(args.eval_signal_summary_json))
        if str(args.eval_signal_summary_json).strip()
        else None
    )
    summary_json = (
        Path(str(args.history_sequence_summary_json))
        if str(args.history_sequence_summary_json).strip()
        else None
    )
    bundle_json = (
        Path(str(args.history_sequence_reporting_bundle_json))
        if str(args.history_sequence_reporting_bundle_json).strip()
        else eval_dir / "history_sequence_reporting_bundle.json"
    )
    metrics = collect_metrics(
        eval_dir,
        days=max(1, int(args.days)),
        eval_signal_summary_json=eval_signal_summary_json,
        history_sequence_summary_json=summary_json,
        history_sequence_reporting_bundle_json=bundle_json,
    )
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
