#!/usr/bin/env python3
"""Generate a release/status-friendly canonical summary from existing stack artifacts.

Reads eval_reporting_index.json and eval_reporting_stack_summary.json, and
normalizes into a minimal release summary with a thin derived release_readiness
signal. Does NOT regenerate bundles, health, index, or public index.
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


def build_release_summary(
    *,
    index_json_path: Optional[Path] = None,
    stack_summary_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ix = load_json_dict(index_json_path) if index_json_path else None
    ss = load_json_dict(stack_summary_json_path) if stack_summary_json_path else None

    ix_d = ix if isinstance(ix, dict) else {}
    ss_d = ss if isinstance(ss, dict) else {}

    stack_status = _safe_str(ss_d.get("status"))
    missing = int(ss_d.get("missing_count", 0) or 0)
    stale = int(ss_d.get("stale_count", 0) or 0)
    mismatch = int(ss_d.get("mismatch_count", 0) or 0)

    if not stack_status or stack_status == "unavailable":
        readiness = "unavailable"
    elif stack_status == "ok" and missing == 0 and stale == 0 and mismatch == 0:
        readiness = "ready"
    else:
        readiness = "degraded"

    return {
        "status": readiness,
        "surface_kind": "eval_reporting_release_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stack_summary_status": stack_status or "unknown",
        "missing_count": missing,
        "stale_count": stale,
        "mismatch_count": mismatch,
        "landing_page_path": _safe_str(ix_d.get("landing_page_html")),
        "static_report_path": _safe_str(ix_d.get("static_report_html")),
        "interactive_report_path": _safe_str(ix_d.get("interactive_report_html")),
        "release_readiness": readiness,
    }


def _render_release_summary_markdown(rs: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Release Summary",
        "",
        f"- Release readiness: `{rs.get('release_readiness', '')}`",
        f"- Generated at: `{rs.get('generated_at', '')}`",
        f"- Stack summary status: `{rs.get('stack_summary_status', '')}`",
        "",
        "## Health Counts",
        "",
        f"- Missing: `{rs.get('missing_count', 0)}`",
        f"- Stale: `{rs.get('stale_count', 0)}`",
        f"- Mismatch: `{rs.get('mismatch_count', 0)}`",
        "",
        "## Artifact Paths",
        "",
        f"- Landing page: `{rs.get('landing_page_path', '')}`",
        f"- Static report: `{rs.get('static_report_path', '')}`",
        f"- Interactive report: `{rs.get('interactive_report_path', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate release/status-friendly eval reporting summary."
    )
    parser.add_argument(
        "--index-json",
        default="reports/eval_history/eval_reporting_index.json",
        help="eval_reporting_index.json path.",
    )
    parser.add_argument(
        "--stack-summary-json",
        default="reports/ci/eval_reporting_stack_summary.json",
        help="eval_reporting_stack_summary.json path.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/ci/eval_reporting_release_summary.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/ci/eval_reporting_release_summary.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rs = build_release_summary(
        index_json_path=Path(args.index_json) if str(args.index_json).strip() else None,
        stack_summary_json_path=Path(args.stack_summary_json) if str(args.stack_summary_json).strip() else None,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rs, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_release_summary_markdown(rs), encoding="utf-8")

    print(f"Release summary JSON: {out_json}")
    print(f"Release summary Markdown: {out_md}")
    print(f"Release readiness: {rs['release_readiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
