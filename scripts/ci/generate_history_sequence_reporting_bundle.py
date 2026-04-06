#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_trend
from scripts import summarize_history_sequence_runs as canonical
from scripts.ci import generate_eval_weekly_summary as weekly
from scripts.ci import generate_history_sequence_compare_report as compare_report


def _materialize_or_load_summary(
    *,
    eval_history_dir: Path,
    report_glob: str,
    summary_json: Path,
) -> Tuple[Dict[str, Any], str]:
    rows = canonical._collect_reports(eval_history_dir, report_glob)
    if rows:
        return (
            canonical._build_summary(rows, eval_history_dir=eval_history_dir, report_glob=report_glob),
            "materialized_from_raw",
        )
    return (
        canonical._load_or_build_summary(
            summary_json, eval_history_dir=eval_history_dir, report_glob=report_glob
        ),
        "loaded_from_existing_summary",
    )


def _build_bundle_markdown(manifest: Dict[str, Any]) -> str:
    lines = [
        "# History Sequence Reporting Bundle",
        "",
        f"- Eval history dir: `{manifest.get('eval_history_dir', '')}`",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Summary source mode: `{manifest.get('summary_source_mode', '')}`",
        f"- Report count: `{manifest.get('report_count', 0)}`",
        f"- Surface group count: `{manifest.get('surface_group_count', 0)}`",
        f"- Best surface key: `{manifest.get('best_surface_key_by_mean_accuracy_overall', '')}`",
        "",
        "## Generated Files",
        "",
        f"- Summary JSON: `{manifest.get('summary_json', '')}`",
        f"- Summary Markdown: `{manifest.get('summary_md', '')}`",
        f"- Compare JSON: `{manifest.get('compare_json', '')}`",
        f"- Compare Markdown: `{manifest.get('compare_md', '')}`",
        f"- Weekly Markdown: `{manifest.get('weekly_md', '')}`",
        f"- Trend directory: `{manifest.get('trend_out_dir', '')}`",
        "",
        "## Trend Outputs",
        "",
    ]
    for path in manifest.get("trend_outputs", []):
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize canonical history-sequence reporting outputs from one bundle entry."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history JSON records.",
    )
    parser.add_argument(
        "--report-glob",
        default="*.json",
        help="Glob pattern used under --eval-history-dir for history-sequence raw rows.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Weekly summary rolling window in days.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Canonical summary JSON path (default: <eval-history-dir>/history_sequence_experiment_summary.json).",
    )
    parser.add_argument(
        "--summary-md",
        default="",
        help="Canonical summary Markdown path (default: <eval-history-dir>/history_sequence_experiment_summary.md).",
    )
    parser.add_argument(
        "--compare-json",
        default="",
        help="Comparison report JSON path (default: <eval-history-dir>/history_sequence_surface_comparison_report.json).",
    )
    parser.add_argument(
        "--compare-md",
        default="",
        help="Comparison report Markdown path (default: <eval-history-dir>/history_sequence_surface_comparison_report.md).",
    )
    parser.add_argument(
        "--weekly-md",
        default="",
        help="Weekly summary Markdown path (default: <eval-history-dir>/weekly_summary.md).",
    )
    parser.add_argument(
        "--trend-out-dir",
        default="",
        help="Trend output directory (default: <eval-history-dir>/plots).",
    )
    parser.add_argument(
        "--bundle-json",
        default="",
        help="Bundle manifest JSON path (default: <eval-history-dir>/history_sequence_reporting_bundle.json).",
    )
    parser.add_argument(
        "--bundle-md",
        default="",
        help="Bundle manifest Markdown path (default: <eval-history-dir>/history_sequence_reporting_bundle.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    summary_json = (
        Path(str(args.summary_json))
        if str(args.summary_json).strip()
        else eval_history_dir / "history_sequence_experiment_summary.json"
    )
    summary_md = (
        Path(str(args.summary_md))
        if str(args.summary_md).strip()
        else eval_history_dir / "history_sequence_experiment_summary.md"
    )
    compare_json = (
        Path(str(args.compare_json))
        if str(args.compare_json).strip()
        else eval_history_dir / "history_sequence_surface_comparison_report.json"
    )
    compare_md = (
        Path(str(args.compare_md))
        if str(args.compare_md).strip()
        else eval_history_dir / "history_sequence_surface_comparison_report.md"
    )
    weekly_md = (
        Path(str(args.weekly_md))
        if str(args.weekly_md).strip()
        else eval_history_dir / "weekly_summary.md"
    )
    trend_out_dir = (
        Path(str(args.trend_out_dir))
        if str(args.trend_out_dir).strip()
        else eval_history_dir / "plots"
    )
    bundle_json = (
        Path(str(args.bundle_json))
        if str(args.bundle_json).strip()
        else eval_history_dir / "history_sequence_reporting_bundle.json"
    )
    bundle_md = (
        Path(str(args.bundle_md))
        if str(args.bundle_md).strip()
        else eval_history_dir / "history_sequence_reporting_bundle.md"
    )

    summary, summary_source_mode = _materialize_or_load_summary(
        eval_history_dir=eval_history_dir,
        report_glob=str(args.report_glob),
        summary_json=summary_json,
    )
    summary_rows = eval_trend.load_history_sequence_rows(
        eval_history_dir,
        history_sequence_summary=summary,
    )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(canonical._build_markdown(summary), encoding="utf-8")

    compare_payload = compare_report._build_report(
        summary_rows,
        eval_history_dir=eval_history_dir,
        report_glob=str(args.report_glob),
    )
    compare_json.parent.mkdir(parents=True, exist_ok=True)
    compare_md.parent.mkdir(parents=True, exist_ok=True)
    compare_json.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_md.write_text(compare_report._build_markdown(compare_payload), encoding="utf-8")

    weekly_metrics = weekly.collect_metrics(
        eval_history_dir,
        days=max(1, int(args.days)),
        history_sequence_summary_json=summary_json,
        history_sequence_summary=summary,
    )
    weekly_text = weekly.build_weekly_markdown(
        metrics=weekly_metrics,
        days=max(1, int(args.days)),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        context={
            "graph2d_blind_status": "unknown",
            "graph2d_blind_accuracy": "n/a",
            "hybrid_blind_status": "unknown",
            "hybrid_blind_accuracy": "n/a",
            "hybrid_blind_gain": "n/a",
            "hybrid_calibration_status": "unknown",
            "hybrid_calibration_gate_status": "unknown",
        },
    )
    weekly_md.parent.mkdir(parents=True, exist_ok=True)
    weekly_md.write_text(weekly_text, encoding="utf-8")

    combined, ocr_only, _ = eval_trend.load_history(eval_history_dir)
    trend_out_dir.mkdir(parents=True, exist_ok=True)
    trend_outputs: List[str] = []
    for path in [
        eval_trend.plot_combined(combined, trend_out_dir),
        eval_trend.plot_ocr_only(ocr_only, trend_out_dir),
        eval_trend.plot_history_sequence(summary_rows, trend_out_dir),
        eval_trend.write_history_sequence_metadata(summary_rows, trend_out_dir),
        eval_trend.plot_history_sequence_by_surface(summary_rows, trend_out_dir),
        eval_trend.write_history_sequence_surface_metadata(summary_rows, trend_out_dir),
    ]:
        if path is not None:
            trend_outputs.append(str(path))

    window = canonical._build_window_summary(summary_rows)
    manifest = {
        "status": "ok",
        "surface_kind": "history_sequence_reporting_bundle",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "summary_source_mode": summary_source_mode,
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "compare_json": str(compare_json),
        "compare_md": str(compare_md),
        "weekly_md": str(weekly_md),
        "trend_out_dir": str(trend_out_dir),
        "trend_outputs": trend_outputs,
        "report_count": int(summary.get("report_count", 0) or 0),
        "surface_group_count": int(window.get("surface_group_count", 0) or 0),
        "best_surface_key_by_mean_accuracy_overall": str(
            window.get("best_surface_key_by_mean_accuracy_overall") or ""
        ),
        "summary_generated_at": str(summary.get("generated_at") or ""),
    }
    bundle_json.parent.mkdir(parents=True, exist_ok=True)
    bundle_md.parent.mkdir(parents=True, exist_ok=True)
    bundle_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_md.write_text(_build_bundle_markdown(manifest), encoding="utf-8")

    print(f"History sequence bundle JSON: {bundle_json}")
    print(f"History sequence bundle Markdown: {bundle_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
