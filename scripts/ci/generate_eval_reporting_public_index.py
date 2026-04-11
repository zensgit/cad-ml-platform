#!/usr/bin/env python3
"""Generate a public-facing discovery surface after Pages deployment.

Reads the eval reporting index and stack summary, combines with the
Pages base URL, and produces a public discovery JSON + Markdown.

Does NOT regenerate landing page, recompute stack summary/health/bundle/index,
or own any new metrics schema.
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


def _join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}" if base and path else base or path


def build_public_index(
    *,
    page_url: str,
    index_json_path: Optional[Path] = None,
    stack_summary_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ix = load_json_dict(index_json_path) if index_json_path else None
    ss = load_json_dict(stack_summary_json_path) if stack_summary_json_path else None

    ix_d = ix if isinstance(ix, dict) else {}
    ss_d = ss if isinstance(ss, dict) else {}

    base = page_url.rstrip("/")

    return {
        "status": "ok" if base else "no_page_url",
        "surface_kind": "eval_reporting_public_index",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_url": base,
        "landing_page_url": _join_url(base, "index.html") if base else "",
        "static_report_url": _join_url(base, "report_static/index.html") if base else "",
        "interactive_report_url": _join_url(base, "report_interactive/index.html") if base else "",
        "stack_summary_status": _safe_str(ss_d.get("status")) or "unknown",
        "missing_count": int(ss_d.get("missing_count", 0) or 0),
        "stale_count": int(ss_d.get("stale_count", 0) or 0),
        "mismatch_count": int(ss_d.get("mismatch_count", 0) or 0),
    }


def _render_public_index_markdown(pi: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Public Index",
        "",
        f"- Status: `{pi.get('status', '')}`",
        f"- Generated at: `{pi.get('generated_at', '')}`",
        f"- Pages URL: `{pi.get('page_url', '')}`",
        "",
        "## Public URLs",
        "",
        f"- Landing page: [{pi.get('landing_page_url', '')}]({pi.get('landing_page_url', '')})",
        f"- Static report: [{pi.get('static_report_url', '')}]({pi.get('static_report_url', '')})",
        f"- Interactive report: [{pi.get('interactive_report_url', '')}]({pi.get('interactive_report_url', '')})",
        "",
        "## Stack Status",
        "",
        f"- Summary status: `{pi.get('stack_summary_status', '')}`",
        f"- Missing: `{pi.get('missing_count', 0)}`",
        f"- Stale: `{pi.get('stale_count', 0)}`",
        f"- Mismatch: `{pi.get('mismatch_count', 0)}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate public-facing eval reporting discovery surface."
    )
    parser.add_argument(
        "--page-url",
        default="",
        help="Pages base URL (e.g. https://user.github.io/repo).",
    )
    parser.add_argument(
        "--index-json",
        default="",
        help="eval_reporting_index.json path.",
    )
    parser.add_argument(
        "--stack-summary-json",
        default="",
        help="eval_reporting_stack_summary.json path.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/ci/eval_reporting_public_index.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/ci/eval_reporting_public_index.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    pi = build_public_index(
        page_url=str(args.page_url).strip(),
        index_json_path=Path(args.index_json) if str(args.index_json).strip() else None,
        stack_summary_json_path=Path(args.stack_summary_json) if str(args.stack_summary_json).strip() else None,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pi, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_public_index_markdown(pi), encoding="utf-8")

    print(f"Public index JSON: {out_json}")
    print(f"Public index Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
