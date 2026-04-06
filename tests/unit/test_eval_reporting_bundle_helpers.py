from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_eval_reporting_bundle_returns_manifest(tmp_path: Path) -> None:
    from scripts.eval_reporting_bundle_helpers import load_eval_reporting_bundle

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_bundle",
            "eval_history_dir": str(history_dir),
        },
    )

    bundle = load_eval_reporting_bundle(history_dir)

    assert isinstance(bundle, dict)
    assert bundle["surface_kind"] == "eval_reporting_bundle"


def test_load_eval_reporting_bundle_returns_none_when_missing(tmp_path: Path) -> None:
    from scripts.eval_reporting_bundle_helpers import load_eval_reporting_bundle

    bundle = load_eval_reporting_bundle(tmp_path / "nonexistent")

    assert bundle is None


def test_load_eval_reporting_assets_follows_bundle_pointers(tmp_path: Path) -> None:
    from scripts.eval_reporting_bundle_helpers import load_eval_reporting_assets

    history_dir = tmp_path / "eval_history"
    es_bundle_path = history_dir / "eval_signal_reporting_bundle.json"
    hs_bundle_path = history_dir / "history_sequence_reporting_bundle.json"

    _write_json(
        es_bundle_path,
        {"status": "ok", "surface_kind": "eval_signal_reporting_bundle", "report_count": 3},
    )
    _write_json(
        hs_bundle_path,
        {"status": "ok", "surface_kind": "history_sequence_reporting_bundle", "report_count": 2},
    )
    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_bundle",
            "eval_signal_bundle_json": str(es_bundle_path),
            "history_sequence_bundle_json": str(hs_bundle_path),
        },
    )

    top, es, hs = load_eval_reporting_assets(history_dir)

    assert isinstance(top, dict)
    assert top["surface_kind"] == "eval_reporting_bundle"
    assert isinstance(es, dict)
    assert es["report_count"] == 3
    assert isinstance(hs, dict)
    assert hs["report_count"] == 2


def test_load_eval_reporting_assets_falls_back_to_default_paths(tmp_path: Path) -> None:
    from scripts.eval_reporting_bundle_helpers import load_eval_reporting_assets

    history_dir = tmp_path / "eval_history"
    _write_json(
        history_dir / "eval_signal_reporting_bundle.json",
        {"status": "ok", "surface_kind": "eval_signal_reporting_bundle", "report_count": 1},
    )
    _write_json(
        history_dir / "history_sequence_reporting_bundle.json",
        {"status": "ok", "surface_kind": "history_sequence_reporting_bundle", "report_count": 1},
    )

    top, es, hs = load_eval_reporting_assets(history_dir)

    assert top is None
    assert isinstance(es, dict)
    assert isinstance(hs, dict)


def test_build_eval_reporting_discovery_context_available(tmp_path: Path) -> None:
    from scripts.eval_reporting_bundle_helpers import build_eval_reporting_discovery_context

    top = {
        "status": "ok",
        "surface_kind": "eval_reporting_bundle",
        "generated_at": "2026-03-30T10:00:00Z",
        "eval_history_dir": str(tmp_path),
        "static_report_html": str(tmp_path / "report_static" / "index.html"),
        "interactive_report_html": str(tmp_path / "report_interactive" / "index.html"),
        "plots_dir": str(tmp_path / "plots"),
        "eval_signal_bundle_json": str(tmp_path / "eval_signal.json"),
        "history_sequence_bundle_json": str(tmp_path / "history_sequence.json"),
    }
    es = {"report_count": 3, "surface_kind": "eval_signal_reporting_bundle"}
    hs = {"report_count": 2, "surface_kind": "history_sequence_reporting_bundle"}

    ctx = build_eval_reporting_discovery_context(top, es, hs)

    assert ctx["available"] is True
    assert ctx["generated_at"] == "2026-03-30T10:00:00Z"
    assert ctx["eval_signal_report_count"] == 3
    assert ctx["history_sequence_report_count"] == 2
    assert "report_static" in ctx["static_report_html"]
    assert "report_interactive" in ctx["interactive_report_html"]


def test_build_eval_reporting_discovery_context_unavailable() -> None:
    from scripts.eval_reporting_bundle_helpers import build_eval_reporting_discovery_context

    ctx = build_eval_reporting_discovery_context(None, None, None)

    assert ctx["available"] is False
    assert ctx["eval_signal_report_count"] == 0
    assert ctx["history_sequence_report_count"] == 0


def test_helper_module_does_not_own_metrics_logic() -> None:
    """The helper module must not define summary/metrics/trend/render functions."""
    import ast
    from scripts import eval_reporting_bundle_helpers as mod

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
    )
    for name in function_names:
        for prefix in forbidden_prefixes:
            assert not name.startswith(prefix), (
                f"Helper must not own metrics/render logic, but defines {name}"
            )
