from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def test_render_dual_summary_with_compare_and_compare_md(tmp_path: Path) -> None:
    from scripts.ci import render_hybrid_superpass_dual_summary as mod

    dual_json = tmp_path / "in" / "dual.json"
    compare_json = tmp_path / "in" / "compare.json"
    compare_md = tmp_path / "in" / "compare.md"
    output_md = tmp_path / "out" / "dual-summary.md"

    _write_json(
        dual_json,
        {
            "overall_exit_code": 1,
            "fail_dispatch_exit_code": 0,
            "success_dispatch_exit_code": 0,
            "compare_exit_code": 1,
            "fail": {
                "run_id": 101,
                "run_url": "https://example.com/runs/101",
                "dispatch_trace_id": "trace-500-fail",
            },
            "success": {
                "run_id": 102,
                "run_url": "https://example.com/runs/102",
                "dispatch_trace_id": "trace-500-success",
            },
        },
    )
    _write_json(
        compare_json,
        {
            "run_id_is_different": True,
            "checks": {
                "fail_expected_failure": True,
                "success_expected_success": True,
                "trace_pair_consistent": True,
            },
            "strict_mode": True,
            "strict_require_distinct_run_ids": True,
            "strict_require_trace_pair": False,
            "strict_failed": True,
        },
    )
    compare_md.parent.mkdir(parents=True, exist_ok=True)
    compare_md.write_text("## Compare Detail\n\n- from compare markdown\n", encoding="utf-8")

    rc = mod.main(
        [
            "--dual-summary-json",
            str(dual_json),
            "--compare-json",
            str(compare_json),
            "--compare-md",
            str(compare_md),
            "--output-md",
            str(output_md),
            "--title",
            "Nightly Dual Summary",
        ]
    )

    assert rc == 0
    markdown = output_md.read_text(encoding="utf-8")
    assert "# Nightly Dual Summary" in markdown
    assert "- overall: `1`" in markdown
    assert "- fail dispatch: `0`" in markdown
    assert "- success dispatch: `0`" in markdown
    assert "- compare: `1`" in markdown
    assert "- run_id_is_different: `true`" in markdown
    assert "- fail_expected_failure: `true`" in markdown
    assert "- success_expected_success: `true`" in markdown
    assert "- trace_pair_consistent: `true`" in markdown
    assert "strict_mode=true" in markdown
    assert "strict_require_distinct_run_ids=true" in markdown
    assert "strict_require_trace_pair=false" in markdown
    assert "- strict_failed: `true`" in markdown
    assert "### fail" in markdown
    assert "- run_id: `101`" in markdown
    assert "- run_url: `https://example.com/runs/101`" in markdown
    assert "- dispatch_trace_id: `trace-500-fail`" in markdown
    assert "### success" in markdown
    assert "- run_id: `102`" in markdown
    assert "- run_url: `https://example.com/runs/102`" in markdown
    assert "- dispatch_trace_id: `trace-500-success`" in markdown
    assert "## Compare Markdown" in markdown
    assert "## Compare Detail" in markdown
    assert "- from compare markdown" in markdown


def test_render_dual_summary_without_compare_inputs(tmp_path: Path) -> None:
    from scripts.ci import render_hybrid_superpass_dual_summary as mod

    dual_json = tmp_path / "in" / "dual.json"
    output_md = tmp_path / "out" / "dual-summary.md"

    _write_json(dual_json, {"overall_exit_code": 0})

    rc = mod.main(
        [
            "--dual-summary-json",
            str(dual_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    markdown = output_md.read_text(encoding="utf-8")
    assert "# Hybrid Superpass Dual Dispatch Summary" in markdown
    assert "## Exit Codes" in markdown
    assert "- overall: `0`" in markdown
    assert "- fail dispatch: `-`" in markdown
    assert "- success dispatch: `-`" in markdown
    assert "- compare: `-`" in markdown
    assert "## Key Checks" in markdown
    assert "- run_id_is_different: `-`" in markdown
    assert "- fail_expected_failure: `-`" in markdown
    assert "- success_expected_success: `-`" in markdown
    assert "- trace_pair_consistent: `-`" in markdown
    assert "- strict_failed: `-`" in markdown
    assert "### fail" in markdown
    assert "### success" in markdown
    assert "## Compare Markdown" not in markdown


def test_render_dual_summary_returns_one_when_dual_json_invalid(tmp_path: Path) -> None:
    from scripts.ci import render_hybrid_superpass_dual_summary as mod

    dual_json = tmp_path / "in" / "dual.json"
    output_md = tmp_path / "out" / "dual-summary.md"

    dual_json.parent.mkdir(parents=True, exist_ok=True)
    dual_json.write_text("{invalid", encoding="utf-8")

    rc = mod.main(
        [
            "--dual-summary-json",
            str(dual_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 1
    assert not output_md.exists()
