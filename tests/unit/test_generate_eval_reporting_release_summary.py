from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_release_readiness_ready_when_all_ok(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_release_summary import build_release_summary

    index_path = tmp_path / "index.json"
    summary_path = tmp_path / "summary.json"
    _write_json(
        index_path,
        {
            "status": "ok",
            "landing_page_html": "reports/eval_history/index.html",
            "static_report_html": "reports/eval_history/report_static/index.html",
            "interactive_report_html": "reports/eval_history/report_interactive/index.html",
        },
    )
    _write_json(
        summary_path,
        {"status": "ok", "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
    )

    rs = build_release_summary(
        index_json_path=index_path,
        stack_summary_json_path=summary_path,
    )

    assert rs["status"] == "ready"
    assert rs["surface_kind"] == "eval_reporting_release_summary"
    assert rs["release_readiness"] == "ready"
    assert rs["stack_summary_status"] == "ok"
    assert rs["missing_count"] == 0
    assert rs["landing_page_path"] == "reports/eval_history/index.html"
    assert rs["static_report_path"] == "reports/eval_history/report_static/index.html"
    assert rs["interactive_report_path"] == "reports/eval_history/report_interactive/index.html"


def test_release_readiness_degraded_when_missing_artifacts(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_release_summary import build_release_summary

    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {"status": "ok", "missing_count": 1, "stale_count": 0, "mismatch_count": 0},
    )

    rs = build_release_summary(stack_summary_json_path=summary_path)

    assert rs["release_readiness"] == "degraded"
    assert rs["status"] == "degraded"
    assert rs["missing_count"] == 1


def test_release_readiness_degraded_when_stack_degraded(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_release_summary import build_release_summary

    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {"status": "degraded", "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
    )

    rs = build_release_summary(stack_summary_json_path=summary_path)

    assert rs["release_readiness"] == "degraded"


def test_release_readiness_unavailable_when_no_summary() -> None:
    from scripts.ci.generate_eval_reporting_release_summary import build_release_summary

    rs = build_release_summary()

    assert rs["release_readiness"] == "unavailable"
    assert rs["status"] == "unavailable"
    assert rs["stack_summary_status"] == "unknown"


def test_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_release_summary as mod

    index_path = tmp_path / "index.json"
    summary_path = tmp_path / "summary.json"
    _write_json(index_path, {"status": "ok"})
    _write_json(
        summary_path,
        {"status": "ok", "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
    )

    out_json = tmp_path / "release.json"
    out_md = tmp_path / "release.md"

    rc = mod.main([
        "--index-json", str(index_path),
        "--stack-summary-json", str(summary_path),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    rs = json.loads(out_json.read_text(encoding="utf-8"))
    assert rs["surface_kind"] == "eval_reporting_release_summary"

    md = out_md.read_text(encoding="utf-8")
    assert "Eval Reporting Release Summary" in md
    assert "Release readiness" in md


def test_helper_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import generate_eval_reporting_release_summary as mod

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
                f"Release summary must not own content logic, but defines {name}"
            )
