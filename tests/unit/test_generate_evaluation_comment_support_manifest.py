from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["generate_evaluation_comment_support_manifest.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def test_build_report_marks_warning_when_some_support_files_are_missing(tmp_path: Path) -> None:
    from scripts.ci import generate_evaluation_comment_support_manifest as mod

    reports_dir = tmp_path / "reports" / "ci"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "workflow_file_health_for_comment.json").write_text(
        json.dumps({"summary": "failed=0/33, mode=yaml_local, fallback=none"}),
        encoding="utf-8",
    )
    (reports_dir / "workflow_inventory_for_comment.md").write_text(
        "# Workflow Inventory Audit\n",
        encoding="utf-8",
    )

    report = mod.build_report(reports_dir=str(reports_dir))

    assert report["overall_status"] == "warning"
    assert report["overall_light"] == "🟡"
    assert report["summary"] == "present=2/11, missing=9, invalid=0"
    assert report["present_count"] == 2
    assert report["missing_count"] == 9
    assert report["invalid_count"] == 0
    assert report["entries"][0]["id"] == "workflow_file_health_json"
    assert report["entries"][0]["summary"] == "failed=0/33, mode=yaml_local, fallback=none"
    assert report["entries"][1]["id"] == "workflow_inventory_json"
    assert report["entries"][1]["present"] is False


def test_build_report_marks_error_when_json_is_invalid(tmp_path: Path) -> None:
    from scripts.ci import generate_evaluation_comment_support_manifest as mod

    reports_dir = tmp_path / "reports" / "ci"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "workflow_file_health_for_comment.json").write_text(
        "{invalid json",
        encoding="utf-8",
    )

    report = mod.build_report(reports_dir=str(reports_dir))

    assert report["overall_status"] == "error"
    assert report["overall_light"] == "🔴"
    assert report["summary"] == "present=1/11, missing=10, invalid=1"
    assert report["entries"][0]["parse_status"] == "error"
    assert "parse_error:" in report["entries"][0]["summary"]


def test_render_markdown_includes_entries_and_summary() -> None:
    from scripts.ci import generate_evaluation_comment_support_manifest as mod

    markdown = mod.render_markdown(
        {
            "overall_status": "warning",
            "overall_light": "🟡",
            "summary": "present=10/11, missing=1, invalid=0",
            "reports_dir": "reports/ci",
            "entries": [
                {
                    "id": "workflow_inventory_json",
                    "path": "workflow_inventory_for_comment.json",
                    "kind": "json",
                    "present": True,
                    "summary": "workflows=33, duplicate=0, missing_required=0, non_unique_required=0",
                },
                {
                    "id": "ci_watch_validation_md",
                    "path": "ci_watch_validation_for_comment.md",
                    "kind": "md",
                    "present": False,
                    "summary": "missing",
                },
            ],
        }
    )

    assert "# Evaluation Comment Support Manifest" in markdown
    assert "overall_status: `warning`" in markdown
    assert "`workflow_inventory_json` [json] `workflow_inventory_for_comment.json`" in markdown
    assert "workflows=33, duplicate=0, missing_required=0, non_unique_required=0" in markdown
    assert "`ci_watch_validation_md` [md] `ci_watch_validation_for_comment.md`: missing" in markdown
    assert "WARN: evaluation comment support bundle is partial." in markdown


def test_build_report_uses_verdict_or_status_fallback_when_summary_missing(tmp_path: Path) -> None:
    from scripts.ci import generate_evaluation_comment_support_manifest as mod

    reports_dir = tmp_path / "reports" / "ci"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "workflow_file_health_for_comment.json").write_text(
        json.dumps({"verdict": "ok"}),
        encoding="utf-8",
    )
    (reports_dir / "workflow_inventory_for_comment.json").write_text(
        json.dumps({"overall_status": "warning"}),
        encoding="utf-8",
    )

    report = mod.build_report(reports_dir=str(reports_dir))

    assert report["entries"][0]["summary"] == "verdict=ok"
    assert report["entries"][1]["summary"] == "status=warning"


def test_main_writes_json_and_markdown(tmp_path: Path, capsys: Any) -> None:
    from scripts.ci import generate_evaluation_comment_support_manifest as mod

    reports_dir = tmp_path / "reports" / "ci"
    output_json = tmp_path / "out" / "manifest.json"
    output_md = tmp_path / "out" / "manifest.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "workflow_file_health_for_comment.json").write_text(
        json.dumps({"summary": "failed=0/33, mode=yaml_local, fallback=none"}),
        encoding="utf-8",
    )

    rc = _invoke_main(
        mod,
        [
            "--reports-dir",
            str(reports_dir),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "warning"
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Evaluation Comment Support Manifest" in rendered
    captured = capsys.readouterr().out
    assert "output_json=" in captured
    assert "output_md=" in captured
