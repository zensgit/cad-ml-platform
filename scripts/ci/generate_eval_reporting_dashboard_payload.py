#!/usr/bin/env python3
"""Generate a dashboard-friendly canonical payload from existing artifacts.

Reads eval_reporting_release_summary.json and eval_reporting_public_index.json,
and normalizes into a minimal external-dashboard-friendly payload.

Does NOT regenerate release summary, public index, stack summary, bundles,
health, or any reports.
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


def build_dashboard_payload(
    *,
    release_summary_json_path: Optional[Path] = None,
    public_index_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    rs = load_json_dict(release_summary_json_path) if release_summary_json_path else None
    pi = load_json_dict(public_index_json_path) if public_index_json_path else None

    rs_d = rs if isinstance(rs, dict) else {}
    pi_d = pi if isinstance(pi, dict) else {}

    readiness = _safe_str(rs_d.get("release_readiness")) or "unavailable"
    stack_status = _safe_str(rs_d.get("stack_summary_status")) or "unknown"
    missing = int(rs_d.get("missing_count", 0) or 0)
    stale = int(rs_d.get("stale_count", 0) or 0)
    mismatch = int(rs_d.get("mismatch_count", 0) or 0)

    landing_url = _safe_str(pi_d.get("landing_page_url"))
    static_url = _safe_str(pi_d.get("static_report_url"))
    interactive_url = _safe_str(pi_d.get("interactive_report_url"))

    public_discovery_ready = bool(landing_url and static_url and interactive_url)

    headline_parts = [f"readiness={readiness}"]
    if missing or stale or mismatch:
        headline_parts.append(f"missing={missing}, stale={stale}, mismatch={mismatch}")
    if public_discovery_ready:
        headline_parts.append("public=ready")
    else:
        headline_parts.append("public=unavailable")
    dashboard_headline = "; ".join(headline_parts)

    return {
        "status": readiness,
        "surface_kind": "eval_reporting_dashboard_payload",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release_readiness": readiness,
        "stack_status": stack_status,
        "missing_count": missing,
        "stale_count": stale,
        "mismatch_count": mismatch,
        "public_landing_page_url": landing_url,
        "public_static_report_url": static_url,
        "public_interactive_report_url": interactive_url,
        "dashboard_headline": dashboard_headline,
        "public_discovery_ready": public_discovery_ready,
    }


def _render_dashboard_payload_markdown(dp: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Dashboard Payload",
        "",
        f"- Headline: `{dp.get('dashboard_headline', '')}`",
        f"- Release readiness: `{dp.get('release_readiness', '')}`",
        f"- Stack status: `{dp.get('stack_status', '')}`",
        f"- Public discovery ready: `{dp.get('public_discovery_ready', False)}`",
        "",
        "## Health Counts",
        "",
        f"- Missing: `{dp.get('missing_count', 0)}`",
        f"- Stale: `{dp.get('stale_count', 0)}`",
        f"- Mismatch: `{dp.get('mismatch_count', 0)}`",
        "",
        "## Public URLs",
        "",
        f"- Landing page: `{dp.get('public_landing_page_url', '')}`",
        f"- Static report: `{dp.get('public_static_report_url', '')}`",
        f"- Interactive report: `{dp.get('public_interactive_report_url', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate dashboard-friendly eval reporting payload."
    )
    parser.add_argument(
        "--release-summary-json",
        default="reports/ci/eval_reporting_release_summary.json",
    )
    parser.add_argument(
        "--public-index-json",
        default="reports/ci/eval_reporting_public_index.json",
    )
    parser.add_argument(
        "--output-json",
        default="reports/ci/eval_reporting_dashboard_payload.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/ci/eval_reporting_dashboard_payload.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dp = build_dashboard_payload(
        release_summary_json_path=Path(args.release_summary_json) if str(args.release_summary_json).strip() else None,
        public_index_json_path=Path(args.public_index_json) if str(args.public_index_json).strip() else None,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dp, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_dashboard_payload_markdown(dp), encoding="utf-8")

    print(f"Dashboard payload JSON: {out_json}")
    print(f"Dashboard payload Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
