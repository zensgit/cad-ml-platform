from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def test_build_comment_body_contains_expected_fields() -> None:
    from scripts.ci import post_soft_mode_smoke_pr_comment as mod

    body = mod.build_comment_body(
        summary={
            "overall_exit_code": 0,
            "dispatch_exit_code": 0,
            "soft_marker_ok": True,
            "restore_ok": True,
            "dispatch": {"run_id": 1234, "run_url": "https://example.invalid/r/1234"},
            "attempts": [
                {
                    "attempt": 1,
                    "dispatch_exit_code": 0,
                    "soft_marker_ok": True,
                    "soft_marker_message": "",
                }
            ],
        },
        title="CAD ML Platform - Soft Mode Smoke",
        commit_sha="abcdef123456",
        updated_at="2026-03-17 10:00:00",
    )
    assert "CAD ML Platform - Soft Mode Smoke" in body
    assert "| Field | Value |" in body
    assert "overall_exit_code" in body
    assert "dispatch_exit_code | 0" in body
    assert "run_id | 1234" in body
    assert "### Attempts" in body
    assert "attempt 1: dispatch_exit_code=0, soft_marker_ok=true" in body
    assert "*Updated: 2026-03-17 10:00:00 UTC*" in body
    assert "*Commit: abcdef1*" in body


def test_main_creates_comment_when_no_existing_bot_comment(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import post_soft_mode_smoke_pr_comment as mod

    summary_json = tmp_path / "soft_smoke.json"
    summary_json.write_text(
        json.dumps(
            {
                "overall_exit_code": 0,
                "dispatch_exit_code": 0,
                "soft_marker_ok": True,
                "restore_ok": True,
                "dispatch": {"run_id": 1234, "run_url": "https://example.invalid/r/1234"},
                "attempts": [{"attempt": 1, "dispatch_exit_code": 0, "soft_marker_ok": True}],
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "result.json"
    commands: list[list[str]] = []

    def _fake_run(
        command: list[str],
        input: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["gh", "api", "repos/zensgit/cad-ml-platform/issues/369/comments"]:
            if "--method" in command:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=json.dumps({"id": 777}),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="[]",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="unexpected")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--pr-number",
            "369",
            "--summary-json",
            str(summary_json),
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["action"] == "create_comment"
    assert payload["created_comment_id"] == 777
    assert any("--method" in cmd and "POST" in cmd for cmd in commands)


def test_main_updates_existing_bot_comment(monkeypatch: Any, tmp_path: Path) -> None:
    from scripts.ci import post_soft_mode_smoke_pr_comment as mod

    summary_json = tmp_path / "soft_smoke.json"
    summary_json.write_text(
        json.dumps(
            {
                "overall_exit_code": 0,
                "dispatch_exit_code": 0,
                "soft_marker_ok": True,
                "restore_ok": True,
                "dispatch": {"run_id": 1234},
                "attempts": [],
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "result_update.json"

    def _fake_run(
        command: list[str],
        input: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["gh", "api", "repos/zensgit/cad-ml-platform/issues/369/comments"]:
            if "--method" in command:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=1,
                    stdout="",
                    stderr="should not create",
                )
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "id": 888,
                            "body": "## CAD ML Platform - Soft Mode Smoke\nold body",
                            "user": {"type": "Bot"},
                        }
                    ]
                ),
                stderr="",
            )
        if command[:3] == ["gh", "api", "repos/zensgit/cad-ml-platform/issues/comments/888"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps({"id": 888}),
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="unexpected")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--pr-number",
            "369",
            "--summary-json",
            str(summary_json),
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["action"] == "update_comment"
    assert payload["updated_comment_id"] == 888


def test_main_dry_run_does_not_call_create_or_update(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from scripts.ci import post_soft_mode_smoke_pr_comment as mod

    summary_json = tmp_path / "soft_smoke.json"
    summary_json.write_text(
        json.dumps({"overall_exit_code": 0, "dispatch_exit_code": 0, "attempts": []}),
        encoding="utf-8",
    )
    output_json = tmp_path / "dry_run_result.json"
    commands: list[list[str]] = []

    def _fake_run(
        command: list[str],
        input: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["gh", "api", "repos/zensgit/cad-ml-platform/issues/369/comments"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="[]",
                stderr="",
            )
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="unexpected")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.main(
        [
            "--repo",
            "zensgit/cad-ml-platform",
            "--pr-number",
            "369",
            "--summary-json",
            str(summary_json),
            "--dry-run",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["action"] == "dry_run_create_comment"
    assert "## CAD ML Platform - Soft Mode Smoke" in payload["body"]
    assert all("POST" not in cmd and "PATCH" not in cmd for cmd in commands)
