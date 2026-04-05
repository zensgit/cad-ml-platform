#!/usr/bin/env python3
"""Summarize the top-level eval reporting stack status for CI/workflow consumption.

Reads bundle, health, and index artifacts and produces a workflow-friendly
JSON + Markdown summary. Does NOT materialize bundles, recompute summaries,
render HTML, or own any new metrics schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_report_data_helpers import load_json_dict


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def build_stack_summary(
    eval_history_dir: Path,
    *,
    refresh_exit_code: int = 0,
    bundle_json_path: Optional[Path] = None,
    health_json_path: Optional[Path] = None,
    index_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle_path = bundle_json_path or (eval_history_dir / "eval_reporting_bundle.json")
    health_path = health_json_path or (eval_history_dir / "eval_reporting_bundle_health_report.json")
    index_path = index_json_path or (eval_history_dir / "eval_reporting_index.json")

    bundle = load_json_dict(bundle_path)
    health = load_json_dict(health_path)
    index = load_json_dict(index_path)

    b = bundle if isinstance(bundle, dict) else {}
    h = health if isinstance(health, dict) else {}
    h_summary = h.get("summary") if isinstance(h.get("summary"), dict) else {}
    ix = index if isinstance(index, dict) else {}

    overall_ok = (
        refresh_exit_code == 0
        and _safe_str(b.get("status")) == "ok"
        and _safe_str(h.get("status")) == "ok"
        and _safe_str(ix.get("status")) == "ok"
    )

    return {
        "status": "ok" if overall_ok else "degraded",
        "surface_kind": "eval_reporting_stack_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "refresh_exit_code": refresh_exit_code,
        "bundle_status": _safe_str(b.get("status")) or "missing",
        "health_status": _safe_str(h.get("status")) or "missing",
        "index_status": _safe_str(ix.get("status")) or "missing",
        "missing_count": int(h_summary.get("missing_count", 0) or 0),
        "stale_count": int(h_summary.get("stale_count", 0) or 0),
        "mismatch_count": int(h_summary.get("mismatch_count", 0) or 0),
        "static_report_html": _safe_str(b.get("static_report_html")),
        "interactive_report_html": _safe_str(b.get("interactive_report_html")),
        "eval_signal_bundle_json": _safe_str(b.get("eval_signal_bundle_json")),
        "history_sequence_bundle_json": _safe_str(b.get("history_sequence_bundle_json")),
    }


def _render_stack_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Stack Summary",
        "",
        f"- Status: `{summary.get('status', '')}`",
        f"- Generated at: `{summary.get('generated_at', '')}`",
        f"- Refresh exit code: `{summary.get('refresh_exit_code', '')}`",
        "",
        "## Component Status",
        "",
        f"- Bundle: `{summary.get('bundle_status', '')}`",
        f"- Health: `{summary.get('health_status', '')}`",
        f"- Index: `{summary.get('index_status', '')}`",
        "",
        "## Health Counts",
        "",
        f"- Missing: `{summary.get('missing_count', 0)}`",
        f"- Stale: `{summary.get('stale_count', 0)}`",
        f"- Mismatch: `{summary.get('mismatch_count', 0)}`",
        "",
        "## Report Paths",
        "",
        f"- Static report: `{summary.get('static_report_html', '')}`",
        f"- Interactive report: `{summary.get('interactive_report_html', '')}`",
        f"- Eval signal bundle: `{summary.get('eval_signal_bundle_json', '')}`",
        f"- History sequence bundle: `{summary.get('history_sequence_bundle_json', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize eval reporting stack status for CI/workflow consumption."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--refresh-exit-code",
        type=int,
        default=0,
        help="Exit code from the refresh step.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Summary Markdown output path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    output_json = (
        Path(str(args.output_json))
        if str(args.output_json).strip()
        else Path("reports/ci/eval_reporting_stack_summary.json")
    )
    output_md = (
        Path(str(args.output_md))
        if str(args.output_md).strip()
        else Path("reports/ci/eval_reporting_stack_summary.md")
    )

    summary = build_stack_summary(
        eval_history_dir,
        refresh_exit_code=args.refresh_exit_code,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_render_stack_summary_markdown(summary), encoding="utf-8")

    print(f"Stack summary JSON: {output_json}")
    print(f"Stack summary Markdown: {output_md}")
    print(f"Status: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
