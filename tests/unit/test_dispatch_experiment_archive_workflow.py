from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


class _Args(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:
        return None


def _command_text(command: Any) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(str(item) for item in command)
    return str(command)


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    main_func = module.main
    params = inspect.signature(main_func).parameters
    try:
        if len(params) == 0:
            old_argv = sys.argv
            try:
                sys.argv = ["dispatch_experiment_archive_workflow.py", *(argv or [])]
                result = main_func()
            finally:
                sys.argv = old_argv
        else:
            result = main_func(list(argv or []))
    except SystemExit as exc:
        return int(exc.code)
    return int(result)


def _patch_parsed_args(monkeypatch: Any, args_obj: _Args) -> None:
    def _fake_parse_args(_self: Any, *_args: Any, **_kwargs: Any) -> _Args:
        return args_obj

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse_args)


def test_resolve_workflow_name_maps_modes() -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    assert mod.resolve_workflow_name("dry-run") == "Experiment Archive Dry Run"
    assert mod.resolve_workflow_name("apply") == "Experiment Archive Apply"


def test_build_workflow_run_command_dry_run_contains_ref_and_shared_inputs() -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    workflow_name = mod.resolve_workflow_name("dry-run")
    command = mod.build_workflow_run_command(
        _Args(
            mode="dry-run",
            ref="main",
            experiments_root="reports/experiments",
            archive_root="reports/archives/experiments",
            keep_latest_days=7,
            today="20260221",
            approval_phrase="",
            dirs_csv="",
            require_exists="true",
        )
    )
    text = _command_text(command)
    assert workflow_name in text
    assert "--ref" in text
    assert "-f" in text
    assert "experiments_root=" in text
    assert "archive_root=" in text
    assert "keep_latest_days=" in text


def test_build_workflow_run_command_apply_contains_approval_and_require_exists() -> (
    None
):
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    workflow_name = mod.resolve_workflow_name("apply")
    command = mod.build_workflow_run_command(
        _Args(
            mode="apply",
            ref="main",
            experiments_root="reports/experiments",
            archive_root="reports/archives/experiments",
            keep_latest_days=7,
            today="20260221",
            approval_phrase="I_UNDERSTAND_DELETE_SOURCE",
            dirs_csv="20260217,20260218",
            require_exists="true",
        )
    )
    text = _command_text(command)
    assert workflow_name in text
    assert "--ref" in text
    assert "approval_phrase=" in text
    assert "I_UNDERSTAND_DELETE_SOURCE" in text
    assert "require_exists=" in text


def test_main_apply_returns_non_zero_when_approval_phrase_mismatch(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            mode="apply",
            ref="main",
            experiments_root="reports/experiments",
            archive_root="reports/archives/experiments",
            keep_latest_days=7,
            today="",
            require_exists=True,
            approval_phrase="WRONG_PHRASE",
            print_only=False,
        ),
    )

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = _invoke_main(mod)
    assert rc != 0


def test_find_latest_dispatched_run_id_returns_match_on_dispatch_event(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    captured: dict[str, Any] = {}
    payload = [
        {"databaseId": 22248985990, "createdAt": "2026-02-21T10:00:00Z"},
        {"databaseId": 22248985989, "createdAt": "2026-02-20T10:00:00Z"},
    ]

    def _fake_run(
        command: Any, *_args: Any, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    run_id = mod.find_latest_dispatched_run_id(
        mod.resolve_workflow_name("dry-run"), "main"
    )
    assert str(run_id) == "22248985990"
    command_text = _command_text(captured["command"])
    assert "--event workflow_dispatch" in command_text
    assert "--branch main" in command_text


def test_find_latest_dispatched_run_id_returns_none_when_no_match(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="[]",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    run_id = mod.find_latest_dispatched_run_id(
        mod.resolve_workflow_name("dry-run"), "main"
    )
    assert run_id is None


def test_main_print_only_does_not_execute_dispatch_or_watch(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            mode="dry-run",
            ref="main",
            experiments_root="reports/experiments",
            archive_root="reports/archives/experiments",
            keep_latest_days=7,
            today="",
            require_exists=True,
            approval_phrase="",
            print_only=True,
            watch=True,
            wait=True,
        ),
    )

    dispatch_called = {"count": 0}
    watch_lookup_called = {"count": 0}
    subprocess_commands: list[str] = []

    def _fake_build_workflow_run_command(*_args: Any, **_kwargs: Any) -> list[str]:
        dispatch_called["count"] += 1
        return ["gh", "workflow", "run", "Experiment Archive Dry Run", "--ref", "main"]

    def _fake_find_latest_dispatched_run_id(*_args: Any, **_kwargs: Any) -> None:
        watch_lookup_called["count"] += 1
        return None

    def _fake_subprocess_run(
        command: Any, *_args: Any, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        subprocess_commands.append(_command_text(command))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        mod, "build_workflow_run_command", _fake_build_workflow_run_command
    )
    monkeypatch.setattr(
        mod, "find_latest_dispatched_run_id", _fake_find_latest_dispatched_run_id
    )
    monkeypatch.setattr(mod.subprocess, "run", _fake_subprocess_run)

    rc = _invoke_main(mod)
    assert rc == 0
    assert dispatch_called["count"] == 1
    assert watch_lookup_called["count"] == 0
    assert all(
        "workflow run" not in item and "run watch" not in item
        for item in subprocess_commands
    )
