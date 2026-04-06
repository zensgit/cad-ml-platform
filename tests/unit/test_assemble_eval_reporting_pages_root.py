from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_html(path: Path, content: str = "<html><body>report</body></html>") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_full_stack(history_dir: Path) -> None:
    _write_html(history_dir / "index.html", "<html><body>landing</body></html>")
    _write_html(history_dir / "report_static" / "index.html", "<html><body>static</body></html>")
    _write_html(history_dir / "report_interactive" / "index.html", "<html><body>interactive</body></html>")
    _write_json(history_dir / "eval_reporting_bundle.json", {"status": "ok"})
    _write_json(history_dir / "eval_reporting_bundle_health_report.json", {"status": "ok"})
    _write_json(history_dir / "eval_reporting_index.json", {"status": "ok"})


def test_assemble_copies_landing_page_as_root_index(tmp_path: Path) -> None:
    from scripts.ci.assemble_eval_reporting_pages_root import assemble

    history_dir = tmp_path / "eval_history"
    pages_root = tmp_path / "eval_pages"
    _setup_full_stack(history_dir)

    rc = assemble(history_dir, pages_root)

    assert rc == 0
    root_index = pages_root / "index.html"
    assert root_index.exists()
    assert "landing" in root_index.read_text(encoding="utf-8")


def test_assemble_copies_static_and_interactive_reports(tmp_path: Path) -> None:
    from scripts.ci.assemble_eval_reporting_pages_root import assemble

    history_dir = tmp_path / "eval_history"
    pages_root = tmp_path / "eval_pages"
    _setup_full_stack(history_dir)

    assemble(history_dir, pages_root)

    assert (pages_root / "report_static" / "index.html").exists()
    assert "static" in (pages_root / "report_static" / "index.html").read_text(encoding="utf-8")
    assert (pages_root / "report_interactive" / "index.html").exists()
    assert "interactive" in (pages_root / "report_interactive" / "index.html").read_text(encoding="utf-8")


def test_assemble_copies_canonical_json_assets(tmp_path: Path) -> None:
    from scripts.ci.assemble_eval_reporting_pages_root import assemble

    history_dir = tmp_path / "eval_history"
    pages_root = tmp_path / "eval_pages"
    _setup_full_stack(history_dir)

    assemble(history_dir, pages_root)

    assert (pages_root / "eval_reporting_bundle.json").exists()
    assert (pages_root / "eval_reporting_bundle_health_report.json").exists()
    assert (pages_root / "eval_reporting_index.json").exists()


def test_assemble_handles_missing_sources(tmp_path: Path) -> None:
    from scripts.ci.assemble_eval_reporting_pages_root import assemble

    history_dir = tmp_path / "empty_history"
    history_dir.mkdir()
    pages_root = tmp_path / "eval_pages"

    rc = assemble(history_dir, pages_root)

    assert rc == 0
    assert pages_root.exists()
    assert not (pages_root / "index.html").exists()


def test_main_cli_interface(tmp_path: Path) -> None:
    from scripts.ci import assemble_eval_reporting_pages_root as mod

    history_dir = tmp_path / "eval_history"
    pages_root = tmp_path / "eval_pages"
    _setup_full_stack(history_dir)

    rc = mod.main([
        "--eval-history-dir", str(history_dir),
        "--pages-root", str(pages_root),
    ])

    assert rc == 0
    assert (pages_root / "index.html").exists()
    assert (pages_root / "report_static" / "index.html").exists()
    assert (pages_root / "report_interactive" / "index.html").exists()


def test_assembler_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import assemble_eval_reporting_pages_root as mod

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
                f"Assembler must not own content logic, but defines {name}"
            )
