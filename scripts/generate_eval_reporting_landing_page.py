#!/usr/bin/env python3
"""Generate a lightweight HTML landing / discovery page for the eval reporting stack.

Reads existing canonical artifacts (index, stack summary, health report) and
renders a single-file static HTML page with navigation links and status badges.

Does NOT materialize bundles, recompute summaries, render charts, or own metrics.
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


def _status_badge(status: str) -> str:
    colors = {
        "ok": "#16a34a",
        "degraded": "#d97706",
        "unhealthy": "#dc2626",
        "missing": "#6b7280",
        "no_bundle": "#6b7280",
    }
    color = colors.get(status, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.85em;">{status}</span>'


def _link_row(label: str, path: str, *, relative_to: Optional[Path] = None) -> str:
    if not path:
        return f"<tr><td>{label}</td><td>{_status_badge('missing')}</td></tr>"
    display = path
    href = path
    if relative_to:
        try:
            href = str(Path(path).relative_to(relative_to))
        except ValueError:
            href = path
        display = href
    return f'<tr><td>{label}</td><td><a href="{href}">{display}</a></td></tr>'


def _health_row(name: str, status: str, detail: str) -> str:
    return f"<tr><td>{name}</td><td>{_status_badge(status)}</td><td>{detail or '-'}</td></tr>"


def load_landing_context(
    eval_history_dir: Path,
    *,
    index_json_path: Optional[Path] = None,
    stack_summary_json_path: Optional[Path] = None,
    health_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ss_path = stack_summary_json_path or Path("reports/ci/eval_reporting_stack_summary.json")
    ix = load_json_dict(index_json_path or (eval_history_dir / "eval_reporting_index.json"))
    ss = load_json_dict(ss_path)
    hr = load_json_dict(health_json_path or (eval_history_dir / "eval_reporting_bundle_health_report.json"))

    return {
        "index": ix if isinstance(ix, dict) else {},
        "stack_summary": ss if isinstance(ss, dict) else {},
        "health_report": hr if isinstance(hr, dict) else {},
        "index_available": isinstance(ix, dict),
        "stack_summary_available": isinstance(ss, dict),
        "health_report_available": isinstance(hr, dict),
        "stack_summary_json_path": str(ss_path),
    }


def render_landing_html(
    ctx: Dict[str, Any],
    *,
    eval_history_dir: Optional[Path] = None,
) -> str:
    ix = ctx.get("index") or {}
    ss = ctx.get("stack_summary") or {}
    hr = ctx.get("health_report") or {}

    overall_status = _safe_str(ss.get("status")) or (_safe_str(ix.get("status")) or "unknown")
    generated_at = _safe_str(ss.get("generated_at")) or _safe_str(ix.get("generated_at")) or ""

    rel = eval_history_dir

    # Build artifact links
    link_rows = []
    link_rows.append(_link_row("Static Report", _safe_str(ix.get("static_report_html")), relative_to=rel))
    link_rows.append(_link_row("Interactive Report", _safe_str(ix.get("interactive_report_html")), relative_to=rel))
    link_rows.append(_link_row("Eval Signal Bundle", _safe_str(ix.get("eval_signal_bundle_json")), relative_to=rel))
    link_rows.append(_link_row("History Sequence Bundle", _safe_str(ix.get("history_sequence_bundle_json")), relative_to=rel))
    link_rows.append(_link_row("Top-Level Bundle", _safe_str(ix.get("eval_reporting_bundle_json")), relative_to=rel))
    link_rows.append(_link_row("Health Report", _safe_str(ix.get("eval_reporting_bundle_health_json")), relative_to=rel))
    link_rows.append(_link_row("Stack Summary", _safe_str(ctx.get("stack_summary_json_path", "")), relative_to=rel))

    # Health checks
    health_rows = []
    if ctx.get("health_report_available"):
        for check in hr.get("checks", []):
            if isinstance(check, dict):
                health_rows.append(_health_row(
                    check.get("name", ""),
                    check.get("status", ""),
                    check.get("detail", ""),
                ))
    else:
        health_rows.append("<tr><td colspan='3'>Health report not available</td></tr>")

    # Summary counts
    missing_count = int(ss.get("missing_count", 0) or 0) if ctx.get("stack_summary_available") else "n/a"
    stale_count = int(ss.get("stale_count", 0) or 0) if ctx.get("stack_summary_available") else "n/a"
    mismatch_count = int(ss.get("mismatch_count", 0) or 0) if ctx.get("stack_summary_available") else "n/a"

    # Missing artifact warnings
    warnings = []
    if not ctx.get("index_available"):
        warnings.append("eval_reporting_index.json is missing")
    if not ctx.get("stack_summary_available"):
        warnings.append("eval_reporting_stack_summary.json is missing")
    if not ctx.get("health_report_available"):
        warnings.append("eval_reporting_bundle_health_report.json is missing")

    warning_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warning_html = f"""
    <div style="background:#fef3c7;border:1px solid #d97706;border-radius:8px;padding:12px 18px;margin-bottom:24px;">
      <strong>Missing Artifacts</strong>
      <ul style="margin:6px 0 0 0;padding-left:20px;">{items}</ul>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Reporting Stack</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 24px; background: #f9fafb; color: #1f2937; }}
  h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
  .meta {{ color: #6b7280; font-size: 0.9em; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
  .counts {{ display: flex; gap: 24px; margin: 12px 0; }}
  .count-item {{ text-align: center; }}
  .count-value {{ font-size: 1.5em; font-weight: 700; }}
  .count-label {{ font-size: 0.8em; color: #6b7280; }}
</style>
</head>
<body>
  <h1>Eval Reporting Stack {_status_badge(overall_status)}</h1>
  <p class="meta">Generated: {generated_at or 'n/a'}</p>
{warning_html}
  <div class="card">
    <h2 style="margin-top:0;">Summary</h2>
    <div class="counts">
      <div class="count-item"><div class="count-value">{missing_count}</div><div class="count-label">Missing</div></div>
      <div class="count-item"><div class="count-value">{stale_count}</div><div class="count-label">Stale</div></div>
      <div class="count-item"><div class="count-value">{mismatch_count}</div><div class="count-label">Mismatch</div></div>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Artifacts</h2>
    <table>
      <thead><tr><th>Name</th><th>Path</th></tr></thead>
      <tbody>
        {"".join(link_rows)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Health Checks</h2>
    <table>
      <thead><tr><th>Name</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>
        {"".join(health_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate eval reporting landing / discovery page."
    )
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history artifacts.",
    )
    parser.add_argument(
        "--index-json", default="",
        help="eval_reporting_index.json path.",
    )
    parser.add_argument(
        "--stack-summary-json", default="",
        help="eval_reporting_stack_summary.json path.",
    )
    parser.add_argument(
        "--health-json", default="",
        help="eval_reporting_bundle_health_report.json path.",
    )
    parser.add_argument(
        "--out", default="",
        help="Output HTML path (default: <eval-history-dir>/index.html).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    output = (
        Path(str(args.out))
        if str(args.out).strip()
        else eval_history_dir / "index.html"
    )

    ctx = load_landing_context(
        eval_history_dir,
        index_json_path=Path(args.index_json) if str(args.index_json).strip() else None,
        stack_summary_json_path=Path(args.stack_summary_json) if str(args.stack_summary_json).strip() else None,
        health_json_path=Path(args.health_json) if str(args.health_json).strip() else None,
    )

    html = render_landing_html(ctx, eval_history_dir=eval_history_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    print(f"Landing page: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
