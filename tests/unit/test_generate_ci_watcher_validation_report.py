from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["generate_ci_watcher_validation_report.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def test_main_generates_success_report_with_explicit_summary(tmp_path: Path) -> None:
    from scripts.ci import generate_ci_watcher_validation_report as mod

    summary_path = tmp_path / "ci" / "watch_commit_abc123def456_summary.json"
    readiness_path = tmp_path / "ci" / "gh_readiness_watch_abc123def456.json"
    soft_smoke_path = tmp_path / "ci" / "evaluation_soft_mode_smoke_summary.json"
    soft_smoke_md_path = tmp_path / "ci" / "evaluation_soft_mode_smoke_summary.md"
    workflow_guardrail_summary_path = tmp_path / "ci" / "workflow_guardrail_summary.json"
    ci_workflow_guardrail_overview_path = tmp_path / "ci" / "ci_workflow_guardrail_overview.json"
    evaluation_comment_support_manifest_path = (
        tmp_path / "ci" / "evaluation_comment_support_manifest.json"
    )
    output_json = tmp_path / "reports" / "ci_watch_validation_summary.json"
    report_dir = tmp_path / "reports"

    _write_json(
        summary_path,
        {
            "requested_sha": "abc123def4567890abc123def4567890abc123de",
            "resolved_sha": "abc123def4567890abc123def4567890abc123de",
            "repo": "zensgit/cad-ml-platform",
            "exit_code": 0,
            "reason": "all_workflows_success",
            "counts": {
                "observed": 2,
                "completed": 2,
                "failed": 0,
                "missing_required": 0,
            },
            "duration_seconds": 12.5,
            "success_conclusions": ["success", "skipped", "neutral"],
            "runs": [
                {"workflow_name": "CI", "conclusion": "success"},
                {"workflow_name": "Code Quality", "conclusion": "success"},
            ],
        },
    )
    _write_json(
        readiness_path,
        {
            "ok": True,
            "checks": [
                {"name": "gh_version", "ok": True, "message": "gh version 2.79.0"},
                {"name": "gh_auth", "ok": True, "message": "gh auth status is ready"},
            ],
        },
    )
    _write_json(
        soft_smoke_path,
        {
            "overall_exit_code": 0,
            "dispatch_exit_code": 0,
            "soft_marker_ok": True,
            "restore_ok": True,
            "max_dispatch_attempts": 2,
            "retry_sleep_seconds": 10,
            "dispatch": {"run_id": 23111551338, "run_url": "https://example.invalid/run"},
            "attempts": [
                {
                    "attempt": 1,
                    "dispatch_exit_code": 0,
                    "soft_marker_ok": True,
                    "soft_marker_message": "",
                }
            ],
        },
    )
    soft_smoke_md_path.write_text(
        "## Evaluation Soft-Mode Smoke\n\n- overall_exit_code: 0\n",
        encoding="utf-8",
    )
    _write_json(
        workflow_guardrail_summary_path,
        {
            "overall_status": "ok",
            "overall_light": "🟢",
            "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
            "workflow_file_health": {
                "status": "ok",
                "summary": "failed=0/33, mode=yaml, fallback=none",
            },
            "workflow_inventory": {
                "status": "ok",
                "summary": "workflows=33, duplicate=0, missing_required=0, non_unique_required=0",
            },
            "workflow_publish_helper": {
                "status": "ok",
                "summary": "checked=33, failed=0, raw=0, missing_comment_helper=0, missing_issue_helper=0",
            },
        },
    )
    _write_json(
        ci_workflow_guardrail_overview_path,
        {
            "overall_status": "ok",
            "overall_light": "🟢",
            "summary": "status=ok, ci_watch=ok, workflow_guardrail=ok",
            "ci_watch": {
                "status": "ok",
                "summary": "reason=all_workflows_success, observed=2, completed=2, failed=0, missing_required=0",
            },
            "workflow_guardrail": {
                "status": "ok",
                "summary": "status=ok, workflow_health=ok, inventory=ok, publish_helper=ok",
            },
        },
    )
    _write_json(
        evaluation_comment_support_manifest_path,
        {
            "overall_status": "ok",
            "overall_light": "🟢",
            "summary": "present=11/11, missing=0, invalid=0",
            "entries": [
                {
                    "id": "workflow_file_health_json",
                    "present": True,
                    "summary": "failed=0/33, mode=yaml_local, fallback=none",
                },
                {
                    "id": "ci_watch_validation_md",
                    "present": True,
                    "summary": "present",
                },
            ],
        },
    )

    rc = _invoke_main(
        mod,
        [
            "--summary-json",
            str(summary_path),
            "--report-dir",
            str(report_dir),
            "--output-json",
            str(output_json),
            "--date",
            "20260225",
            "--report-sha-len",
            "7",
        ],
    )
    assert rc == 0

    output_path = report_dir / "DEV_CI_WATCHER_SAFE_AUTO_SUCCESS_VALIDATION_ABC123D_20260225.md"
    assert output_path.exists()
    assert output_json.exists()
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "PASS." in text
    assert summary_path.as_posix() in text
    assert readiness_path.as_posix() in text
    assert "`repo=zensgit/cad-ml-platform`" in text
    assert "## Soft-Mode Smoke Artifact" in text
    assert soft_smoke_path.as_posix() in text
    assert soft_smoke_md_path.as_posix() in text
    assert "`rendered_markdown=" in text
    assert "`overall_exit_code=0`" in text
    assert "`attempts_total=1`" in text
    assert "attempt#1: dispatch_exit_code=0, soft_marker_ok=True" in text
    assert "## Workflow Guardrail Summary" in text
    assert workflow_guardrail_summary_path.as_posix() in text
    assert "`summary=status=ok, workflow_health=ok, inventory=ok, publish_helper=ok`" in text
    assert "`workflow_file_health.status=ok`" in text
    assert "## CI Workflow Guardrail Overview" in text
    assert ci_workflow_guardrail_overview_path.as_posix() in text
    assert "`summary=status=ok, ci_watch=ok, workflow_guardrail=ok`" in text
    assert "`ci_watch.status=ok`" in text
    assert "## Evaluation Comment Support Manifest" in text
    assert evaluation_comment_support_manifest_path.as_posix() in text
    assert "`summary=present=11/11, missing=0, invalid=0`" in text
    assert "`workflow_file_health_json`: present=True" in text
    assert "No structured failure_details in summary payload." in text
    assert "CI_WATCH_REPO='zensgit/cad-ml-platform'" in text
    assert "CI_WATCH_PRINT_FAILURE_DETAILS=1" in text
    assert payload["verdict"] == "PASS"
    assert payload["summary"] == (
        "verdict=PASS, reason=all_workflows_success, failed=0, missing_required=0, "
        "workflow_guardrail=ok, ci_workflow_overview=ok, comment_support=ok"
    )
    assert payload["summary_path"] == summary_path.as_posix()
    assert payload["workflow_guardrail_summary_path"] == workflow_guardrail_summary_path.as_posix()
    assert payload["ci_workflow_guardrail_overview_path"] == ci_workflow_guardrail_overview_path.as_posix()
    assert (
        payload["evaluation_comment_support_manifest_path"]
        == evaluation_comment_support_manifest_path.as_posix()
    )
    assert payload["sections"]["soft_smoke"]["attempts_total"] == 1
    assert payload["sections"]["evaluation_comment_support_manifest"]["overall_status"] == "ok"


def test_main_auto_picks_latest_summary(tmp_path: Path) -> None:
    from scripts.ci import generate_ci_watcher_validation_report as mod

    summary_dir = tmp_path / "ci"
    report_dir = tmp_path / "reports"
    summary_dir.mkdir(parents=True, exist_ok=True)

    old_sha = "1111111aaaabbbbccccddddeeeeffff00001111"
    new_sha = "2222222aaaabbbbccccddddeeeeffff00002222"
    old_token = old_sha[:12]
    new_token = new_sha[:12]

    old_summary = summary_dir / f"watch_commit_{old_token}_summary.json"
    new_summary = summary_dir / f"watch_commit_{new_token}_summary.json"
    _write_json(
        old_summary,
        {
            "resolved_sha": old_sha,
            "requested_sha": old_sha,
            "exit_code": 0,
            "reason": "all_workflows_success",
            "counts": {"observed": 1, "completed": 1, "failed": 0, "missing_required": 0},
            "runs": [{"workflow_name": "CI", "conclusion": "success"}],
        },
    )
    time.sleep(0.01)
    _write_json(
        new_summary,
        {
            "resolved_sha": new_sha,
            "requested_sha": new_sha,
            "exit_code": 0,
            "reason": "all_workflows_success",
            "counts": {"observed": 1, "completed": 1, "failed": 0, "missing_required": 0},
            "runs": [{"workflow_name": "CI", "conclusion": "success"}],
        },
    )
    _write_json(
        summary_dir / f"gh_readiness_watch_{new_token}.json",
        {"ok": True, "checks": []},
    )

    rc = _invoke_main(
        mod,
        [
            "--summary-dir",
            str(summary_dir),
            "--report-dir",
            str(report_dir),
            "--date",
            "20260225",
        ],
    )
    assert rc == 0

    output_path = report_dir / "DEV_CI_WATCHER_SAFE_AUTO_SUCCESS_VALIDATION_2222222_20260225.md"
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert new_sha in text
    assert old_sha not in text


def test_main_generates_fail_report_when_summary_is_not_green(tmp_path: Path) -> None:
    from scripts.ci import generate_ci_watcher_validation_report as mod

    summary_path = tmp_path / "ci" / "watch_commit_abcdef123456_summary.json"
    report_dir = tmp_path / "reports"
    _write_json(
        summary_path,
        {
            "resolved_sha": "abcdef1234567890abcdef1234567890abcdef12",
            "requested_sha": "abcdef1234567890abcdef1234567890abcdef12",
            "repo": "zensgit/cad-ml-platform",
            "exit_code": 1,
            "reason": "workflow_failed",
            "counts": {"observed": 1, "completed": 1, "failed": 1, "missing_required": 0},
            "failed_workflows": ["Code Quality"],
            "failure_details": [
                {
                    "run_id": 12345,
                    "workflow_name": "Code Quality",
                    "conclusion": "failure",
                    "failed_jobs": ["lint"],
                    "failed_steps": ["lint :: flake8 (failure)"],
                    "url": "https://example.com/runs/12345",
                }
            ],
            "runs": [{"workflow_name": "Code Quality", "conclusion": "failure"}],
        },
    )

    rc = _invoke_main(
        mod,
        [
            "--summary-json",
            str(summary_path),
            "--report-dir",
            str(report_dir),
            "--date",
            "20260225",
        ],
    )
    assert rc == 0

    output_path = report_dir / "DEV_CI_WATCHER_SAFE_AUTO_VALIDATION_ABCDEF1_20260225.md"
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "FAIL." in text
    assert "does not satisfy release-gate CI criteria." in text
    assert "Not found (inferred readiness json is missing)." in text
    assert "## Soft-Mode Smoke Artifact" in text
    assert "Not found (inferred soft-smoke summary json is missing)." in text
    assert "## Workflow Guardrail Summary" in text
    assert "Not found (inferred workflow guardrail summary json is missing)." in text
    assert "## CI Workflow Guardrail Overview" in text
    assert "Not found (inferred ci workflow guardrail overview json is missing)." in text
    assert "## Evaluation Comment Support Manifest" in text
    assert "Not found (inferred evaluation comment support manifest json is missing)." in text
    assert "Failure Details" in text
    assert "Code Quality (run=12345, conclusion=failure)" in text
    assert "failed_jobs: lint" in text
    assert "failed_steps: lint :: flake8 (failure)" in text
    assert "CI_WATCH_REPO='zensgit/cad-ml-platform'" in text


def test_main_accepts_explicit_soft_smoke_markdown_path(tmp_path: Path) -> None:
    from scripts.ci import generate_ci_watcher_validation_report as mod

    summary_path = tmp_path / "ci" / "watch_commit_abc123def456_summary.json"
    soft_smoke_path = tmp_path / "ci" / "soft_smoke.json"
    soft_smoke_md_path = tmp_path / "custom" / "soft_smoke.md"
    report_dir = tmp_path / "reports"

    _write_json(
        summary_path,
        {
            "requested_sha": "abc123def4567890abc123def4567890abc123de",
            "resolved_sha": "abc123def4567890abc123def4567890abc123de",
            "repo": "zensgit/cad-ml-platform",
            "exit_code": 0,
            "reason": "all_workflows_success",
            "counts": {
                "observed": 1,
                "completed": 1,
                "failed": 0,
                "missing_required": 0,
            },
            "runs": [{"workflow_name": "CI", "conclusion": "success"}],
        },
    )
    _write_json(
        soft_smoke_path,
        {
            "overall_exit_code": 0,
            "dispatch_exit_code": 0,
            "soft_marker_ok": True,
            "restore_ok": True,
            "attempts": [],
        },
    )
    soft_smoke_md_path.parent.mkdir(parents=True, exist_ok=True)
    soft_smoke_md_path.write_text("# explicit soft smoke md\n", encoding="utf-8")

    rc = _invoke_main(
        mod,
        [
            "--summary-json",
            str(summary_path),
            "--soft-smoke-summary-json",
            str(soft_smoke_path),
            "--soft-smoke-summary-md",
            str(soft_smoke_md_path),
            "--report-dir",
            str(report_dir),
            "--date",
            "20260225",
        ],
    )
    assert rc == 0

    output_path = report_dir / "DEV_CI_WATCHER_SAFE_AUTO_SUCCESS_VALIDATION_ABC123D_20260225.md"
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert soft_smoke_md_path.as_posix() in text
