from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["check_gh_actions_ready.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def test_main_all_checks_pass_and_writes_json(tmp_path: Any, monkeypatch: Any) -> None:
    from scripts.ci import check_gh_actions_ready as mod

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command == ["gh", "--version"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="gh version 2.79.0\n",
                stderr="",
            )
        if command == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Logged in\n",
                stderr="",
            )
        if command == ["gh", "run", "list", "--limit", "1"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="[]\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    output_path = tmp_path / "gh-ready.json"
    rc = _invoke_main(mod, ["--json-out", str(output_path)])
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert [check["name"] for check in payload["checks"]] == [
        "gh_version",
        "gh_auth",
        "gh_actions_api",
    ]


def test_main_reports_auth_failure(monkeypatch: Any, capsys: Any) -> None:
    from scripts.ci import check_gh_actions_ready as mod

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command == ["gh", "--version"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="gh version\n",
                stderr="",
            )
        if command == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout=(
                    "X Failed to log in to github.com account zensgit (default)\n"
                    "- The token in default is invalid.\n"
                ),
                stderr="",
            )
        if command == ["gh", "run", "list", "--limit", "1"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="[]\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = _invoke_main(mod)
    assert rc == 1
    output = capsys.readouterr().out
    assert "gh auth is not ready" in output
    assert "gh auth login -h github.com" in output


def test_main_reports_actions_api_connectivity_failure(
    monkeypatch: Any, capsys: Any
) -> None:
    from scripts.ci import check_gh_actions_ready as mod

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command == ["gh", "--version"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="gh version\n",
                stderr="",
            )
        if command == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Logged in\n",
                stderr="",
            )
        if command == ["gh", "run", "list", "--limit", "1"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr=(
                    "error connecting to api.github.com\n"
                    "check your internet connection or https://githubstatus.com\n"
                ),
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = _invoke_main(mod)
    assert rc == 1
    output = capsys.readouterr().out
    assert "cannot access GitHub Actions API" in output
    assert "error connecting to api.github.com" in output


def test_extract_error_prefers_keyword_lines() -> None:
    from scripts.ci import check_gh_actions_ready as mod

    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="line-a\nline-b\n",
        stderr="error connecting to api.github.com\n",
    )
    message = mod._extract_error(result, "fallback")
    assert message == "error connecting to api.github.com"


def test_main_skip_actions_api_bypasses_run_list(tmp_path: Any, monkeypatch: Any) -> None:
    from scripts.ci import check_gh_actions_ready as mod

    called_commands: list[list[str]] = []

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        called_commands.append(command)
        if command == ["gh", "--version"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="gh version\n",
                stderr="",
            )
        if command == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="Logged in\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    output_path = tmp_path / "gh-ready-skip.json"
    rc = _invoke_main(mod, ["--skip-actions-api", "--json-out", str(output_path)])
    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["skip_actions_api"] is True
    assert payload["checks"][-1]["name"] == "gh_actions_api"
    assert payload["checks"][-1]["message"] == "skipped by --skip-actions-api"
    assert ["gh", "run", "list", "--limit", "1"] not in called_commands
