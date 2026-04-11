from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["generate_workflow_guardrail_summary.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def test_build_report_marks_all_green_when_inputs_are_healthy() -> None:
    from scripts.ci import generate_workflow_guardrail_summary as mod

    report = mod.build_report(
        workflow_file_health_payload={
            "count": 33,
            "failed_count": 0,
            "mode_used": "auto",
            "fallback_reason": "none",
        },
        workflow_inventory_payload={
            "workflow_count": 33,
            "duplicate_name_count": 0,
            "missing_required_count": 0,
            "non_unique_required_count": 0,
        },
        workflow_publish_helper_payload={
            "checked_count": 33,
            "failed_count": 0,
            "raw_publish_violation_count": 0,
            "missing_comment_helper_import_count": 0,
            "missing_issue_helper_import_count": 0,
        },
    )

    assert report["overall_status"] == "ok"
    assert report["overall_light"] == "🟢"
    assert report["summary"] == "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok"
    assert report["workflow_file_health"]["summary"] == "failed=0/33, mode=auto, fallback=none"
    assert report["workflow_inventory"]["summary"] == (
        "workflows=33, duplicate=0, missing_required=0, non_unique_required=0"
    )
    assert report["workflow_publish_helper"]["summary"] == (
        "checked=33, failed=0, raw=0, missing_comment_helper=0, missing_issue_helper=0"
    )


def test_render_markdown_includes_sections() -> None:
    from scripts.ci import generate_workflow_guardrail_summary as mod

    markdown = mod.render_markdown(
        {
            "overall_status": "error",
            "overall_light": "🔴",
            "summary": "status=error, workflow_health=ok, inventory=error, publish_helper=ok",
            "workflow_file_health": {
                "light": "🟢",
                "summary": "failed=0/33, mode=auto, fallback=none",
            },
            "workflow_inventory": {
                "light": "🔴",
                "summary": "workflows=33, duplicate=1, missing_required=0, non_unique_required=0",
            },
            "workflow_publish_helper": {
                "light": "🟢",
                "summary": "checked=33, failed=0, raw=0, missing_comment_helper=0, missing_issue_helper=0",
            },
        }
    )

    assert "# Workflow Guardrail Summary" in markdown
    assert "overall_status: `error`" in markdown
    assert "workflow_file_health: 🟢 failed=0/33, mode=auto, fallback=none" in markdown
    assert "workflow_inventory: 🔴 workflows=33, duplicate=1" in markdown
    assert "workflow_publish_helper: 🟢 checked=33, failed=0" in markdown


def test_main_writes_json_and_markdown(tmp_path: Path, capsys: Any) -> None:
    from scripts.ci import generate_workflow_guardrail_summary as mod

    workflow_file_health = tmp_path / "workflow_file_health.json"
    workflow_inventory = tmp_path / "workflow_inventory.json"
    workflow_publish_helper = tmp_path / "workflow_publish_helper.json"
    output_json = tmp_path / "guardrail_summary.json"
    output_md = tmp_path / "guardrail_summary.md"

    workflow_file_health.write_text(
        json.dumps({"count": 10, "failed_count": 0, "mode_used": "auto", "fallback_reason": "none"}),
        encoding="utf-8",
    )
    workflow_inventory.write_text(
        json.dumps({"workflow_count": 10, "duplicate_name_count": 0, "missing_required_count": 0, "non_unique_required_count": 0}),
        encoding="utf-8",
    )
    workflow_publish_helper.write_text(
        json.dumps({"checked_count": 10, "failed_count": 0, "raw_publish_violation_count": 0, "missing_comment_helper_import_count": 0, "missing_issue_helper_import_count": 0}),
        encoding="utf-8",
    )

    rc = _invoke_main(
        mod,
        [
            "--workflow-file-health-json",
            str(workflow_file_health),
            "--workflow-inventory-json",
            str(workflow_inventory),
            "--workflow-publish-helper-json",
            str(workflow_publish_helper),
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
    assert "# Workflow Guardrail Summary" in rendered
    captured = capsys.readouterr().out
    assert "output_json=" in captured
    assert "output_md=" in captured
