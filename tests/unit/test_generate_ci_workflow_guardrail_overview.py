from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["generate_ci_workflow_guardrail_overview.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def test_build_report_marks_green_when_both_inputs_are_green() -> None:
    from scripts.ci import generate_ci_workflow_guardrail_overview as mod

    report = mod.build_report(
        ci_watch_payload={
            "requested_sha": "abc123def4567890abc123def4567890abc123de",
            "resolved_sha": "abc123def4567890abc123def4567890abc123de",
            "exit_code": 0,
            "reason": "all_workflows_success",
            "counts": {"observed": 3, "completed": 3, "failed": 0, "missing_required": 0},
        },
        workflow_guardrail_payload={
            "overall_status": "ok",
            "overall_light": "🟢",
            "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
        },
    )

    assert report["overall_status"] == "ok"
    assert report["overall_light"] == "🟢"
    assert report["summary"] == "status=ok, ci_watch=ok, workflow_guardrail=ok"
    assert report["ci_watch"]["summary"] == (
        "reason=all_workflows_success, observed=3, completed=3, failed=0, missing_required=0"
    )
    assert report["workflow_guardrail"]["summary"] == (
        "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok"
    )


def test_render_markdown_includes_key_sections() -> None:
    from scripts.ci import generate_ci_workflow_guardrail_overview as mod

    markdown = mod.render_markdown(
        {
            "overall_status": "error",
            "overall_light": "🔴",
            "summary": "status=error, ci_watch=error, workflow_guardrail=ok",
            "sha": "abc1234",
            "ci_watch": {
                "status": "error",
                "light": "🔴",
                "summary": "reason=workflow_failed, observed=3, completed=3, failed=1, missing_required=0",
                "reason": "workflow_failed",
                "resolved_sha": "abc1234",
            },
            "workflow_guardrail": {
                "status": "ok",
                "light": "🟢",
                "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
            },
        }
    )

    assert "# CI Workflow Guardrail Overview" in markdown
    assert "overall_status: `error`" in markdown
    assert "## CI Watch" in markdown
    assert "reason=workflow_failed, observed=3, completed=3, failed=1, missing_required=0" in markdown
    assert "## Workflow Guardrail" in markdown
    assert "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok" in markdown
    assert "FAIL: at least one of watcher or workflow guardrails is degraded." in markdown


def test_main_writes_json_and_markdown(tmp_path: Path, capsys: Any) -> None:
    from scripts.ci import generate_ci_workflow_guardrail_overview as mod

    watch_summary = tmp_path / "ci" / "watch_commit_abc123def456_summary.json"
    workflow_guardrail = tmp_path / "ci" / "workflow_guardrail_summary.json"
    output_json = tmp_path / "reports" / "overview.json"
    output_md = tmp_path / "reports" / "overview.md"

    watch_summary.parent.mkdir(parents=True, exist_ok=True)
    watch_summary.write_text(
        json.dumps(
            {
                "requested_sha": "abc123def4567890abc123def4567890abc123de",
                "resolved_sha": "abc123def4567890abc123def4567890abc123de",
                "exit_code": 0,
                "reason": "all_workflows_success",
                "counts": {"observed": 2, "completed": 2, "failed": 0, "missing_required": 0},
            }
        ),
        encoding="utf-8",
    )
    workflow_guardrail.write_text(
        json.dumps(
            {
                "overall_status": "ok",
                "overall_light": "🟢",
                "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
            }
        ),
        encoding="utf-8",
    )

    rc = _invoke_main(
        mod,
        [
            "--ci-watch-summary-json",
            str(watch_summary),
            "--workflow-guardrail-json",
            str(workflow_guardrail),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "ok"
    rendered = output_md.read_text(encoding="utf-8")
    assert "# CI Workflow Guardrail Overview" in rendered
    captured = capsys.readouterr().out
    assert "ci_watch_summary_json=" in captured
    assert "output_json=" in captured
    assert "output_md=" in captured
