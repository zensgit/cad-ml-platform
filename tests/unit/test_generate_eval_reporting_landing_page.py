from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_landing_page_renders_with_all_artifacts(tmp_path: Path) -> None:
    from scripts import generate_eval_reporting_landing_page as mod

    history_dir = tmp_path / "eval_history"
    ci_dir = tmp_path / "ci"

    _write_json(
        history_dir / "eval_reporting_index.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_index",
            "generated_at": "2026-03-30T12:00:00Z",
            "eval_reporting_bundle_json": str(history_dir / "eval_reporting_bundle.json"),
            "eval_reporting_bundle_health_json": str(history_dir / "eval_reporting_bundle_health_report.json"),
            "eval_signal_bundle_json": str(history_dir / "eval_signal_reporting_bundle.json"),
            "history_sequence_bundle_json": str(history_dir / "history_sequence_reporting_bundle.json"),
            "static_report_html": str(history_dir / "report_static" / "index.html"),
            "interactive_report_html": str(history_dir / "report_interactive" / "index.html"),
        },
    )
    _write_json(
        ci_dir / "eval_reporting_stack_summary.json",
        {
            "status": "ok",
            "generated_at": "2026-03-30T12:00:00Z",
            "missing_count": 0,
            "stale_count": 0,
            "mismatch_count": 0,
        },
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "ok",
            "checks": [
                {"name": "root_bundle", "status": "ok", "detail": ""},
                {"name": "eval_signal_bundle", "status": "ok", "detail": ""},
            ],
        },
    )

    out = tmp_path / "landing.html"
    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--stack-summary-json", str(ci_dir / "eval_reporting_stack_summary.json"),
        "--out", str(out),
    ])

    assert rc == 0
    assert out.exists()

    html = out.read_text(encoding="utf-8")
    assert "Eval Reporting Stack" in html
    assert "Static Report" in html
    assert "Interactive Report" in html
    assert "report_static" in html
    assert "report_interactive" in html
    assert "eval_signal_reporting_bundle" in html
    assert "history_sequence_reporting_bundle" in html
    assert "eval_reporting_bundle.json" in html
    assert "Health Checks" in html
    assert "root_bundle" in html
    assert "Stack Summary" in html
    assert "eval_reporting_stack_summary" in html


def test_landing_page_shows_missing_when_no_artifacts(tmp_path: Path) -> None:
    from scripts import generate_eval_reporting_landing_page as mod

    out = tmp_path / "landing.html"
    rc = mod.main([
        "--eval-history-dir", str(tmp_path / "nonexistent"),
        "--out", str(out),
    ])

    assert rc == 0
    assert out.exists()

    html = out.read_text(encoding="utf-8")
    assert "Eval Reporting Stack" in html
    assert "Missing Artifacts" in html
    assert "eval_reporting_index.json is missing" in html
    assert "missing" in html.lower()


def test_landing_page_links_static_and_interactive(tmp_path: Path) -> None:
    from scripts import generate_eval_reporting_landing_page as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_index.json",
        {
            "status": "ok",
            "static_report_html": str(history_dir / "report_static" / "index.html"),
            "interactive_report_html": str(history_dir / "report_interactive" / "index.html"),
        },
    )

    out = tmp_path / "landing.html"
    mod.main(["--eval-history-dir", str(history_dir), "--out", str(out)])

    html = out.read_text(encoding="utf-8")
    assert "report_static" in html
    assert "report_interactive" in html
    assert "<a " in html


def test_landing_page_default_output_is_index_html(tmp_path: Path) -> None:
    from scripts import generate_eval_reporting_landing_page as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok"},
    )

    mod.main(["--eval-history-dir", str(history_dir)])

    assert (history_dir / "index.html").exists()


def test_landing_page_shows_health_checks(tmp_path: Path) -> None:
    from scripts import generate_eval_reporting_landing_page as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok"},
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "unhealthy",
            "checks": [
                {"name": "root_bundle", "status": "ok", "detail": ""},
                {"name": "eval_signal_bundle", "status": "missing", "detail": "file does not exist"},
            ],
        },
    )

    out = tmp_path / "landing.html"
    mod.main(["--eval-history-dir", str(history_dir), "--out", str(out)])

    html = out.read_text(encoding="utf-8")
    assert "eval_signal_bundle" in html
    assert "file does not exist" in html


def test_landing_renderer_does_not_own_metrics_logic() -> None:
    import ast
    from scripts import generate_eval_reporting_landing_page as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden_prefixes = (
        "_build_summary", "_build_window", "_collect_reports",
        "plot_", "_mean_from", "build_weekly", "materialize",
    )
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Landing renderer must not own metrics/bundle logic, but defines {name}"
            )
