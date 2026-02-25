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
    report_dir = tmp_path / "reports"

    _write_json(
        summary_path,
        {
            "requested_sha": "abc123def4567890abc123def4567890abc123de",
            "resolved_sha": "abc123def4567890abc123def4567890abc123de",
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

    rc = _invoke_main(
        mod,
        [
            "--summary-json",
            str(summary_path),
            "--report-dir",
            str(report_dir),
            "--date",
            "20260225",
            "--report-sha-len",
            "7",
        ],
    )
    assert rc == 0

    output_path = report_dir / "DEV_CI_WATCHER_SAFE_AUTO_SUCCESS_VALIDATION_ABC123D_20260225.md"
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "PASS." in text
    assert summary_path.as_posix() in text
    assert readiness_path.as_posix() in text


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
            "exit_code": 1,
            "reason": "workflow_failed",
            "counts": {"observed": 1, "completed": 1, "failed": 1, "missing_required": 0},
            "failed_workflows": ["Code Quality"],
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
