from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html><body>report</body></html>", encoding="utf-8")


def _setup_healthy_bundle(history_dir: Path) -> None:
    es_path = history_dir / "eval_signal_reporting_bundle.json"
    hs_path = history_dir / "history_sequence_reporting_bundle.json"
    static_html = history_dir / "report_static" / "index.html"
    interactive_html = history_dir / "report_interactive" / "index.html"
    plots_dir = history_dir / "plots"

    _write_json(es_path, {"status": "ok", "surface_kind": "eval_signal_reporting_bundle"})
    _write_json(hs_path, {"status": "ok", "surface_kind": "history_sequence_reporting_bundle"})
    _write_html(static_html)
    _write_html(interactive_html)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_bundle",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_history_dir": str(history_dir),
            "eval_signal_bundle_json": str(es_path),
            "history_sequence_bundle_json": str(hs_path),
            "static_report_html": str(static_html),
            "interactive_report_html": str(interactive_html),
            "plots_dir": str(plots_dir),
        },
    )


def test_health_check_all_ok(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)

    report = run_health_checks(history_dir)

    assert report["status"] == "ok"
    assert report["surface_kind"] == "eval_reporting_bundle_health_report"
    assert report["summary"]["ok"] is True
    assert report["summary"]["missing_count"] == 0
    assert report["summary"]["stale_count"] == 0
    assert report["summary"]["mismatch_count"] == 0
    assert len(report["missing_artifacts"]) == 0
    assert len(report["stale_artifacts"]) == 0
    assert len(report["mismatch_artifacts"]) == 0


def test_health_check_missing_root_bundle(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    report = run_health_checks(tmp_path / "nonexistent")

    assert report["status"] == "unhealthy"
    assert report["summary"]["missing_count"] >= 1
    assert "root_bundle" in report["missing_artifacts"]


def test_health_check_missing_sub_bundle(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)
    # Remove one sub-bundle
    (history_dir / "eval_signal_reporting_bundle.json").unlink()

    report = run_health_checks(history_dir)

    assert report["status"] == "unhealthy"
    assert "eval_signal_bundle" in report["missing_artifacts"]


def test_health_check_missing_report(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)
    (history_dir / "report_static" / "index.html").unlink()

    report = run_health_checks(history_dir)

    assert report["status"] == "unhealthy"
    assert "static_report" in report["missing_artifacts"]


def test_health_check_stale_bundle(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)

    # Use a future "now" to make everything appear stale
    future = datetime.now(timezone.utc) + timedelta(hours=500)
    report = run_health_checks(
        history_dir,
        max_root_age_hours=24.0,
        max_sub_bundle_age_hours=24.0,
        max_report_age_hours=24.0,
        now=future,
    )

    assert report["status"] == "unhealthy"
    assert report["summary"]["stale_count"] >= 1
    assert len(report["stale_artifacts"]) >= 1


def test_health_check_pointer_mismatch(tmp_path: Path) -> None:
    from scripts.ci.check_eval_reporting_bundle_health import run_health_checks

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)
    # Overwrite eval_signal bundle with wrong surface_kind
    _write_json(
        history_dir / "eval_signal_reporting_bundle.json",
        {"status": "ok", "surface_kind": "wrong_kind"},
    )

    report = run_health_checks(history_dir)

    assert report["status"] == "unhealthy"
    assert "eval_signal_bundle" in report["mismatch_artifacts"]


def test_health_check_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import check_eval_reporting_bundle_health as mod

    history_dir = tmp_path / "eval_history"
    _setup_healthy_bundle(history_dir)
    out_json = tmp_path / "health.json"
    out_md = tmp_path / "health.md"

    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["surface_kind"] == "eval_reporting_bundle_health_report"

    md_text = out_md.read_text(encoding="utf-8")
    assert "Eval Reporting Bundle Health Report" in md_text
    assert "root_bundle" in md_text


def test_health_check_main_returns_nonzero_on_unhealthy(tmp_path: Path) -> None:
    from scripts.ci import check_eval_reporting_bundle_health as mod

    rc = mod.main([
        "--eval-history-dir", str(tmp_path / "nonexistent"),
        "--output-json", str(tmp_path / "health.json"),
        "--output-md", str(tmp_path / "health.md"),
    ])

    assert rc != 0
    report = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert report["status"] == "unhealthy"


def test_health_checker_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import check_eval_reporting_bundle_health as mod

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
                f"Health checker must not own metrics/render/materialization logic, but defines {name}"
            )
