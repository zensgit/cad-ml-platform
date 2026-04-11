#!/usr/bin/env python3
"""Materialize canonical eval-signal reporting outputs from one bundle entry."""
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
from scripts import summarize_eval_signal_runs as canonical
from scripts.ci import generate_eval_weekly_summary as weekly


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
        "# Eval Signal Reporting Bundle",
        "",
        f"- Eval history dir: `{manifest.get('eval_history_dir', '')}`",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Summary source mode: `{manifest.get('summary_source_mode', '')}`",
        f"- Report count: `{manifest.get('report_count', 0)}`",
        f"- Combined reports: `{manifest.get('combined_report_count', 0)}`",
        f"- OCR reports: `{manifest.get('ocr_report_count', 0)}`",
        f"- Hybrid blind reports: `{manifest.get('hybrid_blind_report_count', 0)}`",
        "",
        "## Generated Files",
        "",
        f"- Summary JSON: `{manifest.get('summary_json', '')}`",
        f"- Summary Markdown: `{manifest.get('summary_md', '')}`",
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
        description="Materialize canonical eval-signal reporting outputs from one bundle entry."
    )
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
        "--days",
        type=int,
        default=7,
        help="Weekly summary rolling window in days.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Canonical summary JSON path (default: <eval-history-dir>/eval_signal_experiment_summary.json).",
    )
    parser.add_argument(
        "--summary-md",
        default="",
        help="Canonical summary Markdown path (default: <eval-history-dir>/eval_signal_experiment_summary.md).",
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
        help="Bundle manifest JSON path (default: <eval-history-dir>/eval_signal_reporting_bundle.json).",
    )
    parser.add_argument(
        "--bundle-md",
        default="",
        help="Bundle manifest Markdown path (default: <eval-history-dir>/eval_signal_reporting_bundle.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    summary_json = (
        Path(str(args.summary_json))
        if str(args.summary_json).strip()
        else eval_history_dir / "eval_signal_experiment_summary.json"
    )
    summary_md = (
        Path(str(args.summary_md))
        if str(args.summary_md).strip()
        else eval_history_dir / "eval_signal_experiment_summary.md"
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
        else eval_history_dir / "eval_signal_reporting_bundle.json"
    )
    bundle_md = (
        Path(str(args.bundle_md))
        if str(args.bundle_md).strip()
        else eval_history_dir / "eval_signal_reporting_bundle.md"
    )

    summary, summary_source_mode = _materialize_or_load_summary(
        eval_history_dir=eval_history_dir,
        report_glob=str(args.report_glob),
        summary_json=summary_json,
    )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(canonical._build_markdown(summary), encoding="utf-8")

    weekly_metrics = weekly.collect_metrics(
        eval_history_dir,
        days=max(1, int(args.days)),
        eval_signal_summary_json=summary_json,
        eval_signal_summary=summary,
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

    combined, ocr_only, _ = eval_trend.load_history(
        eval_history_dir,
        eval_signal_summary=summary,
    )
    trend_out_dir.mkdir(parents=True, exist_ok=True)
    trend_outputs: List[str] = []
    for path in [
        eval_trend.plot_combined(combined, trend_out_dir),
        eval_trend.plot_ocr_only(ocr_only, trend_out_dir),
    ]:
        if path is not None:
            trend_outputs.append(str(path))

    report_counts = summary.get("report_counts") if isinstance(summary.get("report_counts"), dict) else {}
    latest_combined = (
        summary.get("latest_combined_run") if isinstance(summary.get("latest_combined_run"), dict) else {}
    )
    latest_ocr = (
        summary.get("latest_ocr_run") if isinstance(summary.get("latest_ocr_run"), dict) else {}
    )
    latest_hybrid = (
        summary.get("latest_hybrid_blind_run")
        if isinstance(summary.get("latest_hybrid_blind_run"), dict)
        else {}
    )
    manifest: Dict[str, Any] = {
        "status": "ok",
        "surface_kind": "eval_signal_reporting_bundle",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "summary_source_mode": summary_source_mode,
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "weekly_md": str(weekly_md),
        "trend_out_dir": str(trend_out_dir),
        "trend_outputs": trend_outputs,
        "report_count": int(summary.get("report_count", 0) or 0),
        "combined_report_count": int(report_counts.get("combined", 0) or 0),
        "ocr_report_count": int(report_counts.get("ocr", 0) or 0),
        "hybrid_blind_report_count": int(report_counts.get("hybrid_blind", 0) or 0),
        "summary_generated_at": str(summary.get("generated_at") or ""),
        "latest_combined_timestamp": str(latest_combined.get("timestamp") or ""),
        "latest_ocr_timestamp": str(latest_ocr.get("timestamp") or ""),
        "latest_hybrid_blind_timestamp": str(latest_hybrid.get("timestamp") or ""),
    }

    bundle_json.parent.mkdir(parents=True, exist_ok=True)
    bundle_md.parent.mkdir(parents=True, exist_ok=True)
    bundle_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_md.write_text(_build_bundle_markdown(manifest), encoding="utf-8")

    print(f"Eval signal bundle JSON: {bundle_json}")
    print(f"Eval signal bundle Markdown: {bundle_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
