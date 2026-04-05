from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_index_populates_discovery_fields_from_bundle(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_index as mod

    history_dir = tmp_path / "eval_history"
    es_path = history_dir / "eval_signal_reporting_bundle.json"
    hs_path = history_dir / "history_sequence_reporting_bundle.json"
    static_html = history_dir / "report_static" / "index.html"
    interactive_html = history_dir / "report_interactive" / "index.html"
    plots_dir = history_dir / "plots"

    _write_json(
        history_dir / "eval_reporting_bundle.json",
        {
            "status": "ok",
            "surface_kind": "eval_reporting_bundle",
            "eval_signal_bundle_json": str(es_path),
            "history_sequence_bundle_json": str(hs_path),
            "static_report_html": str(static_html),
            "interactive_report_html": str(interactive_html),
            "plots_dir": str(plots_dir),
        },
    )

    out_json = tmp_path / "index.json"
    out_md = tmp_path / "index.md"

    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    index = json.loads(out_json.read_text(encoding="utf-8"))
    assert index["status"] == "ok"
    assert index["surface_kind"] == "eval_reporting_index"
    assert "generated_at" in index
    assert index["eval_signal_bundle_json"] == str(es_path)
    assert index["history_sequence_bundle_json"] == str(hs_path)
    assert index["static_report_html"] == str(static_html)
    assert index["interactive_report_html"] == str(interactive_html)
    assert index["plots_dir"] == str(plots_dir)
    assert "eval_reporting_bundle_json" in index
    assert "eval_reporting_bundle_health_json" in index
    assert "landing_page_html" in index
    assert "index.html" in index["landing_page_html"]

    md_text = out_md.read_text(encoding="utf-8")
    assert "Eval Reporting Index" in md_text


def test_index_returns_no_bundle_status_when_missing(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_index as mod

    out_json = tmp_path / "index.json"
    rc = mod.main([
        "--eval-history-dir", str(tmp_path / "nonexistent"),
        "--output-json", str(out_json),
        "--output-md", str(tmp_path / "index.md"),
    ])

    assert rc == 0
    index = json.loads(out_json.read_text(encoding="utf-8"))
    assert index["status"] == "no_bundle"
    assert index["eval_signal_bundle_json"] == ""


def test_index_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import generate_eval_reporting_index as mod

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
                f"Index must not own metrics/render logic, but defines {name}"
            )
