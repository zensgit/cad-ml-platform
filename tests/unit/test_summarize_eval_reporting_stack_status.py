from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_stack_summary_all_ok(tmp_path: Path) -> None:
    from scripts.ci.summarize_eval_reporting_stack_status import build_stack_summary

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_bundle",
            "static_report_html": str(history_dir / "report_static" / "index.html"),
            "interactive_report_html": str(history_dir / "report_interactive" / "index.html"),
            "eval_signal_bundle_json": str(history_dir / "eval_signal.json"),
            "history_sequence_bundle_json": str(history_dir / "history_sequence.json"),
        },
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "ok",
            "summary": {"ok": True, "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
        },
    )
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok", "surface_kind": "eval_reporting_index"},
    )

    summary = build_stack_summary(history_dir, refresh_exit_code=0)

    assert summary["status"] == "ok"
    assert summary["surface_kind"] == "eval_reporting_stack_summary"
    assert summary["refresh_exit_code"] == 0
    assert summary["bundle_status"] == "ok"
    assert summary["health_status"] == "ok"
    assert summary["index_status"] == "ok"
    assert summary["missing_count"] == 0
    assert summary["stale_count"] == 0
    assert summary["mismatch_count"] == 0
    assert "report_static" in summary["static_report_html"]
    assert "report_interactive" in summary["interactive_report_html"]


def test_build_stack_summary_degraded_on_refresh_failure(tmp_path: Path) -> None:
    from scripts.ci.summarize_eval_reporting_stack_status import build_stack_summary

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {"status": "ok", "surface_kind": "eval_reporting_bundle"},
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "ok",
            "summary": {"ok": True, "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
        },
    )
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok"},
    )

    summary = build_stack_summary(history_dir, refresh_exit_code=1)

    assert summary["status"] == "degraded"
    assert summary["refresh_exit_code"] == 1


def test_build_stack_summary_degraded_on_missing_bundle(tmp_path: Path) -> None:
    from scripts.ci.summarize_eval_reporting_stack_status import build_stack_summary

    summary = build_stack_summary(tmp_path / "nonexistent", refresh_exit_code=0)

    assert summary["status"] == "degraded"
    assert summary["bundle_status"] == "missing"


def test_build_stack_summary_propagates_health_counts(tmp_path: Path) -> None:
    from scripts.ci.summarize_eval_reporting_stack_status import build_stack_summary

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {"status": "ok", "surface_kind": "eval_reporting_bundle"},
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "unhealthy",
            "summary": {"ok": False, "missing_count": 1, "stale_count": 2, "mismatch_count": 0},
        },
    )
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok"},
    )

    summary = build_stack_summary(history_dir, refresh_exit_code=0)

    assert summary["status"] == "degraded"
    assert summary["missing_count"] == 1
    assert summary["stale_count"] == 2


def test_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import summarize_eval_reporting_stack_status as mod

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {"status": "ok", "surface_kind": "eval_reporting_bundle"},
    )
    _write_json(
        history_dir / "eval_reporting_bundle_health_report.json",
        {
            "status": "ok",
            "summary": {"ok": True, "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
        },
    )
    _write_json(
        history_dir / "eval_reporting_index.json",
        {"status": "ok"},
    )

    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"

    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["surface_kind"] == "eval_reporting_stack_summary"
    assert summary["status"] == "ok"

    md_text = out_md.read_text(encoding="utf-8")
    assert "Eval Reporting Stack Summary" in md_text


def test_helper_does_not_own_metrics_or_render_logic() -> None:
    import ast
    from scripts.ci import summarize_eval_reporting_stack_status as mod

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
        "materialize",
    )
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Summary helper must not own metrics/render logic, but defines {name}"
            )
