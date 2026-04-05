#!/usr/bin/env python3
"""Generate a webhook delivery request / policy from the dashboard payload.

Reads only eval_reporting_dashboard_payload.json and produces a delivery
request payload suitable for an external webhook sender.

Does NOT read release summary, public index, stack summary, or webhook
export directly. Does NOT send HTTP requests or manage queues.
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

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
WEBHOOK_EVENT_TYPE = "eval_reporting.updated"
INGESTION_SCHEMA_VERSION = "1.0.0"


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def build_delivery_request(
    *,
    dashboard_payload_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dp = load_json_dict(dashboard_payload_json_path) if dashboard_payload_json_path else None
    dp_d = dp if isinstance(dp, dict) else {}

    readiness = _safe_str(dp_d.get("release_readiness")) or "unavailable"
    stack_status = _safe_str(dp_d.get("stack_status")) or "unknown"
    headline = _safe_str(dp_d.get("dashboard_headline")) or f"readiness={readiness}"
    landing_url = _safe_str(dp_d.get("public_landing_page_url"))
    static_url = _safe_str(dp_d.get("public_static_report_url"))
    interactive_url = _safe_str(dp_d.get("public_interactive_report_url"))
    missing = int(dp_d.get("missing_count", 0) or 0)
    stale = int(dp_d.get("stale_count", 0) or 0)
    mismatch = int(dp_d.get("mismatch_count", 0) or 0)

    delivery_allowed = bool(dp_d)

    request_body: Dict[str, Any] = {
        "webhook_event_type": WEBHOOK_EVENT_TYPE,
        "ingestion_schema_version": INGESTION_SCHEMA_VERSION,
        "release_readiness": readiness,
        "stack_status": stack_status,
        "dashboard_headline": headline,
        "missing_count": missing,
        "stale_count": stale,
        "mismatch_count": mismatch,
        "landing_page_url": landing_url,
        "static_report_url": static_url,
        "interactive_report_url": interactive_url,
    }

    return {
        "status": readiness,
        "surface_kind": "eval_reporting_webhook_delivery_request",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "webhook_event_type": WEBHOOK_EVENT_TYPE,
        "ingestion_schema_version": INGESTION_SCHEMA_VERSION,
        "release_readiness": readiness,
        "stack_status": stack_status,
        "dashboard_headline": headline,
        "missing_count": missing,
        "stale_count": stale,
        "mismatch_count": mismatch,
        "landing_page_url": landing_url,
        "static_report_url": static_url,
        "interactive_report_url": interactive_url,
        "delivery_target_kind": "external_webhook",
        "delivery_method": "POST",
        "delivery_policy": "disabled_by_default",
        "delivery_allowed": delivery_allowed,
        "delivery_requires_explicit_enable": True,
        "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "request_body_json": json.dumps(request_body, ensure_ascii=False),
        "source_dashboard_payload_surface_kind": _safe_str(dp_d.get("surface_kind")) or "unknown",
    }


def _render_delivery_request_markdown(dr: Dict[str, Any]) -> str:
    lines = [
        "# Eval Reporting Webhook Delivery Request",
        "",
        f"- Webhook Event: `{dr.get('webhook_event_type', '')}`",
        f"- Delivery Policy: `{dr.get('delivery_policy', '')}`",
        f"- Delivery Allowed: `{dr.get('delivery_allowed', False)}`",
        f"- Delivery Method: `{dr.get('delivery_method', '')}`",
        f"- Delivery Target: `{dr.get('delivery_target_kind', '')}`",
        f"- Release readiness: `{dr.get('release_readiness', '')}`",
        f"- Timeout: `{dr.get('request_timeout_seconds', 30)}s`",
        "",
        "## Public URLs",
        "",
        f"- Landing Page: `{dr.get('landing_page_url', '')}`",
        f"- Static Report: `{dr.get('static_report_url', '')}`",
        f"- Interactive Report: `{dr.get('interactive_report_url', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate webhook delivery request from dashboard payload."
    )
    parser.add_argument(
        "--dashboard-payload-json",
        default="reports/ci/eval_reporting_dashboard_payload.json",
    )
    parser.add_argument(
        "--output-json",
        default="reports/ci/eval_reporting_webhook_delivery_request.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/ci/eval_reporting_webhook_delivery_request.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dr = build_delivery_request(
        dashboard_payload_json_path=Path(args.dashboard_payload_json) if str(args.dashboard_payload_json).strip() else None,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dr, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_delivery_request_markdown(dr), encoding="utf-8")

    print(f"Delivery request JSON: {out_json}")
    print(f"Delivery request Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
