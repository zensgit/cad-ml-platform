from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_delivery_request_with_full_dashboard_payload(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_webhook_delivery_request import build_delivery_request

    dp_path = tmp_path / "dp.json"
    _write_json(dp_path, {
        "surface_kind": "eval_reporting_dashboard_payload",
        "release_readiness": "ready",
        "stack_status": "ok",
        "dashboard_headline": "readiness=ready; public=ready",
        "missing_count": 0, "stale_count": 0, "mismatch_count": 0,
        "public_landing_page_url": "https://example.com/index.html",
        "public_static_report_url": "https://example.com/report_static/index.html",
        "public_interactive_report_url": "https://example.com/report_interactive/index.html",
    })

    dr = build_delivery_request(dashboard_payload_json_path=dp_path)

    assert dr["status"] == "ready"
    assert dr["surface_kind"] == "eval_reporting_webhook_delivery_request"
    assert dr["webhook_event_type"] == "eval_reporting.updated"
    assert dr["ingestion_schema_version"] == "1.0.0"
    assert dr["delivery_target_kind"] == "external_webhook"
    assert dr["delivery_method"] == "POST"
    assert dr["delivery_policy"] == "disabled_by_default"
    assert dr["delivery_allowed"] is True
    assert dr["delivery_requires_explicit_enable"] is True
    assert dr["request_timeout_seconds"] == 30
    assert dr["landing_page_url"] == "https://example.com/index.html"
    assert dr["source_dashboard_payload_surface_kind"] == "eval_reporting_dashboard_payload"

    body = json.loads(dr["request_body_json"])
    assert body["webhook_event_type"] == "eval_reporting.updated"
    assert body["release_readiness"] == "ready"


def test_delivery_request_unavailable_without_input() -> None:
    from scripts.ci.generate_eval_reporting_webhook_delivery_request import build_delivery_request

    dr = build_delivery_request()

    assert dr["status"] == "unavailable"
    assert dr["delivery_allowed"] is False
    assert dr["delivery_requires_explicit_enable"] is True
    assert dr["delivery_policy"] == "disabled_by_default"


def test_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_webhook_delivery_request as mod

    dp_path = tmp_path / "dp.json"
    _write_json(dp_path, {
        "surface_kind": "eval_reporting_dashboard_payload",
        "release_readiness": "ready",
        "public_landing_page_url": "https://example.com/index.html",
        "public_static_report_url": "https://example.com/static/index.html",
        "public_interactive_report_url": "https://example.com/interactive/index.html",
    })

    out_json = tmp_path / "dr.json"
    out_md = tmp_path / "dr.md"

    rc = mod.main([
        "--dashboard-payload-json", str(dp_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    dr = json.loads(out_json.read_text(encoding="utf-8"))
    assert dr["surface_kind"] == "eval_reporting_webhook_delivery_request"

    md = out_md.read_text(encoding="utf-8")
    assert "Webhook Event" in md
    assert "Delivery Policy" in md
    assert "Release readiness" in md
    assert "Landing Page" in md
    assert "Static Report" in md
    assert "Interactive Report" in md


def test_delivery_request_only_reads_dashboard_payload() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "ci" / "generate_eval_reporting_webhook_delivery_request.py"
    ).read_text(encoding="utf-8")

    assert "generate_eval_reporting_webhook_export" not in source
    assert "generate_eval_reporting_release_summary" not in source
    assert "generate_eval_reporting_public_index" not in source
    assert "summarize_eval_reporting_stack_status" not in source
