from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_public_index_with_page_url(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_public_index import build_public_index

    index_path = tmp_path / "index.json"
    summary_path = tmp_path / "summary.json"
    _write_json(index_path, {"status": "ok"})
    _write_json(
        summary_path,
        {"status": "ok", "missing_count": 0, "stale_count": 0, "mismatch_count": 0},
    )

    pi = build_public_index(
        page_url="https://user.github.io/repo",
        index_json_path=index_path,
        stack_summary_json_path=summary_path,
    )

    assert pi["status"] == "ok"
    assert pi["surface_kind"] == "eval_reporting_public_index"
    assert pi["page_url"] == "https://user.github.io/repo"
    assert pi["landing_page_url"] == "https://user.github.io/repo/index.html"
    assert pi["static_report_url"] == "https://user.github.io/repo/report_static/index.html"
    assert pi["interactive_report_url"] == "https://user.github.io/repo/report_interactive/index.html"
    assert pi["stack_summary_status"] == "ok"
    assert pi["missing_count"] == 0


def test_build_public_index_without_page_url() -> None:
    from scripts.ci.generate_eval_reporting_public_index import build_public_index

    pi = build_public_index(page_url="")

    assert pi["status"] == "no_page_url"
    assert pi["landing_page_url"] == ""
    assert pi["static_report_url"] == ""


def test_build_public_index_propagates_health_counts(tmp_path: Path) -> None:
    from scripts.ci.generate_eval_reporting_public_index import build_public_index

    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {"status": "degraded", "missing_count": 1, "stale_count": 2, "mismatch_count": 0},
    )

    pi = build_public_index(
        page_url="https://example.com",
        stack_summary_json_path=summary_path,
    )

    assert pi["stack_summary_status"] == "degraded"
    assert pi["missing_count"] == 1
    assert pi["stale_count"] == 2


def test_main_writes_json_and_md(tmp_path: Path) -> None:
    from scripts.ci import generate_eval_reporting_public_index as mod

    out_json = tmp_path / "public.json"
    out_md = tmp_path / "public.md"

    rc = mod.main([
        "--page-url", "https://user.github.io/repo",
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ])

    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()

    pi = json.loads(out_json.read_text(encoding="utf-8"))
    assert pi["surface_kind"] == "eval_reporting_public_index"

    md = out_md.read_text(encoding="utf-8")
    assert "Eval Reporting Public Index" in md
    assert "report_static" in md
    assert "report_interactive" in md


def test_public_index_does_not_own_metrics_logic() -> None:
    import ast
    from scripts.ci import generate_eval_reporting_public_index as mod

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
                f"Public index must not own content logic, but defines {name}"
            )
