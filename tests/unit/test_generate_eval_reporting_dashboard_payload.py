from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_dashboard_payload_ready_with_public_urls(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_dashboard_payload import build_dashboard_payload

    rs_path = tmp_path / "release.json"
    pi_path = tmp_path / "public.json"
    _write_json(rs_path, {
        "release_readiness": "ready",
        "stack_summary_status": "ok",
        "missing_count": 0, "stale_count": 0, "mismatch_count": 0,
    })
    _write_json(pi_path, {
        "landing_page_url": "https://example.com/index.html",
        "static_report_url": "https://example.com/report_static/index.html",
        "interactive_report_url": "https://example.com/report_interactive/index.html",
    })

    dp = build_dashboard_payload(
        release_summary_json_path=rs_path,
        public_index_json_path=pi_path,
    )

    assert dp["status"] == "ready"
    assert dp["surface_kind"] == "eval_reporting_dashboard_payload"
    assert dp["release_readiness"] == "ready"
    assert dp["stack_status"] == "ok"
    assert dp["public_discovery_ready"] is True
    assert dp["public_landing_page_url"] == "https://example.com/index.html"
    assert dp["public_static_report_url"] == "https://example.com/report_static/index.html"
    assert "dashboard_headline" in dp
    assert "readiness=ready" in dp["dashboard_headline"]
    assert "public=ready" in dp["dashboard_headline"]


def test_dashboard_payload_unavailable_without_inputs() -> None:
    from scripts.ci.generate_eval_reporting_dashboard_payload import build_dashboard_payload

    dp = build_dashboard_payload()

    assert dp["status"] == "unavailable"
    assert dp["release_readiness"] == "unavailable"
    assert dp["public_discovery_ready"] is False


def test_dashboard_payload_degraded_with_missing_count(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_dashboard_payload import build_dashboard_payload

    rs_path = tmp_path / "release.json"
    _write_json(rs_path, {
        "release_readiness": "degraded",
        "stack_summary_status": "ok",
        "missing_count": 1, "stale_count": 0, "mismatch_count": 0,
    })

    dp = build_dashboard_payload(release_summary_json_path=rs_path)

    assert dp["status"] == "degraded"
    assert dp["missing_count"] == 1
    assert "missing=1" in dp["dashboard_headline"]


def test_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_dashboard_payload as mod

    rs_path = tmp_path / "release.json"
    _write_json(rs_path, {
        "release_readiness": "ready",
        "stack_summary_status": "ok",
        "missing_count": 0, "stale_count": 0, "mismatch_count": 0,
    })

    out_json = tmp_path / "dp.json"
    out_md = tmp_path / "dp.md"

    rc = mod.main([
        "--release-summary-json", str(rs_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    dp = json.loads(out_json.read_text(encoding="utf-8"))
    assert dp["surface_kind"] == "eval_reporting_dashboard_payload"

    md = out_md.read_text(encoding="utf-8")
    assert "Dashboard Payload" in md


def test_helper_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import generate_eval_reporting_dashboard_payload as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden_prefixes = (
        "_build_summary", "_build_window", "_collect_reports",
        "plot_", "_mean_from", "generate_html", "build_weekly",
        "materialize", "render_landing",
    )
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Dashboard payload must not own content logic, but defines {name}"
            )
