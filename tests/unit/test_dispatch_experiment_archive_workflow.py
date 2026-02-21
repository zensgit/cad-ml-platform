from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


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


def _call_wait_for_new_dispatched_run_id(
    module: Any,
    *,
    workflow_name: str,
    ref: str,
    known_run_ids: set[int],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> int | None:
    wait_func = module.wait_for_new_dispatched_run_id
    params = inspect.signature(wait_func).parameters
    kwargs: dict[str, Any] = {}

    for name, param in params.items():
        if name == "workflow_name":
            kwargs[name] = workflow_name
        elif name == "ref":
            kwargs[name] = ref
        elif name == "known_run_ids":
            kwargs[name] = known_run_ids
        elif name == "timeout_seconds":
            kwargs[name] = timeout_seconds
        elif name == "poll_interval_seconds":
            kwargs[name] = poll_interval_seconds
        elif param.default is inspect._empty:
            raise AssertionError(
                f"wait_for_new_dispatched_run_id has unsupported required arg: {name}"
            )

    return wait_func(**kwargs)


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

    subprocess_calls = {"count": 0}

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        subprocess_calls["count"] += 1
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    rc = _invoke_main(mod)
    assert rc != 0
    assert subprocess_calls["count"] == 0


def test_check_gh_ready_returns_false_when_gh_binary_missing(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    run_calls = {"count": 0}

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls["count"] += 1
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    is_ready, _reason = mod.check_gh_ready()
    assert is_ready is False
    assert run_calls["count"] >= 1


def test_check_gh_ready_returns_false_when_gh_auth_fails(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    returncodes = iter([0, 1])

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=next(returncodes),
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    is_ready, _reason = mod.check_gh_ready()
    assert is_ready is False


def test_check_gh_ready_returns_true_when_checks_pass(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    is_ready, reason = mod.check_gh_ready()
    assert is_ready is True
    assert reason == ""


def test_list_dispatched_run_ids_parses_json_list(monkeypatch: Any) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    payload = [
        {"databaseId": 22248985990},
        {"databaseId": "22248985989"},
    ]

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    run_ids = mod.list_dispatched_run_ids(mod.resolve_workflow_name("dry-run"), "main")
    assert run_ids == [22248985990, 22248985989]


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, "[]"),
        (0, "{not-json"),
    ],
)
def test_list_dispatched_run_ids_returns_empty_on_invalid_json_or_nonzero(
    monkeypatch: Any,
    returncode: int,
    stdout: str,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    run_ids = mod.list_dispatched_run_ids(mod.resolve_workflow_name("dry-run"), "main")
    assert run_ids == []


def test_wait_for_new_dispatched_run_id_returns_new_id_before_timeout(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    responses = [
        [101, 102],
        [101, 102, 103],
    ]

    def _fake_list_dispatched_run_ids(*_args: Any, **_kwargs: Any) -> list[int]:
        if responses:
            return responses.pop(0)
        return [101, 102, 103]

    monkeypatch.setattr(mod, "list_dispatched_run_ids", _fake_list_dispatched_run_ids)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    run_id = _call_wait_for_new_dispatched_run_id(
        mod,
        workflow_name=mod.resolve_workflow_name("dry-run"),
        ref="main",
        known_run_ids={101, 102},
        timeout_seconds=5,
        poll_interval_seconds=0.01,
    )
    assert run_id == 103


def test_wait_for_new_dispatched_run_id_returns_none_on_timeout(
    monkeypatch: Any,
) -> None:
    from scripts.ci import dispatch_experiment_archive_workflow as mod

    def _fake_list_dispatched_run_ids(*_args: Any, **_kwargs: Any) -> list[int]:
        return [101, 102]

    clock = {"value": 0.0}

    def _fake_time() -> float:
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr(mod, "list_dispatched_run_ids", _fake_list_dispatched_run_ids)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mod.time, "time", _fake_time)

    run_id = _call_wait_for_new_dispatched_run_id(
        mod,
        workflow_name=mod.resolve_workflow_name("dry-run"),
        ref="main",
        known_run_ids={101, 102},
        timeout_seconds=2,
        poll_interval_seconds=0.01,
    )
    assert run_id is None


def test_main_watch_uses_known_run_ids_wait_and_watch_run(monkeypatch: Any) -> None:
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
            print_only=False,
            watch=True,
            wait_timeout_seconds=120,
            poll_interval_seconds=1,
        ),
    )

    calls = {
        "list_dispatched_run_ids": 0,
        "wait_for_new_dispatched_run_id": 0,
        "watch_run": 0,
    }
    captured: dict[str, Any] = {}

    def _fake_build_workflow_run_command(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["gh", "workflow", "run", "Experiment Archive Dry Run", "--ref", "main"]

    def _fake_list_dispatched_run_ids(*_args: Any, **_kwargs: Any) -> list[int]:
        calls["list_dispatched_run_ids"] += 1
        return [100, 101]

    def _fake_wait_for_new_dispatched_run_id(*args: Any, **kwargs: Any) -> int | None:
        calls["wait_for_new_dispatched_run_id"] += 1
        if "known_run_ids" in kwargs:
            captured["known_run_ids"] = kwargs["known_run_ids"]
        elif len(args) >= 3:
            captured["known_run_ids"] = args[2]
        return 102

    def _fake_watch_run(run_id: int) -> int:
        calls["watch_run"] += 1
        captured["watch_run_id"] = run_id
        return 7

    def _fake_subprocess_run(
        _command: Any, *_args: Any, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        mod, "build_workflow_run_command", _fake_build_workflow_run_command
    )
    monkeypatch.setattr(mod, "list_dispatched_run_ids", _fake_list_dispatched_run_ids)
    monkeypatch.setattr(
        mod,
        "wait_for_new_dispatched_run_id",
        _fake_wait_for_new_dispatched_run_id,
    )
    monkeypatch.setattr(mod, "watch_run", _fake_watch_run)
    monkeypatch.setattr(mod.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))

    rc = _invoke_main(mod)
    assert rc == 7
    assert calls["list_dispatched_run_ids"] == 1
    assert calls["wait_for_new_dispatched_run_id"] == 1
    assert calls["watch_run"] == 1
    assert set(captured["known_run_ids"]) == {100, 101}
    assert captured["watch_run_id"] == 102


def test_main_watch_returns_non_zero_when_wait_returns_none(monkeypatch: Any) -> None:
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
            print_only=False,
            watch=True,
            wait_timeout_seconds=120,
            poll_interval_seconds=1,
        ),
    )

    watch_calls = {"count": 0}

    def _fake_build_workflow_run_command(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["gh", "workflow", "run", "Experiment Archive Dry Run", "--ref", "main"]

    def _fake_wait_for_new_dispatched_run_id(*_args: Any, **_kwargs: Any) -> int | None:
        return None

    def _fake_watch_run(_run_id: int) -> int:
        watch_calls["count"] += 1
        return 0

    def _fake_subprocess_run(
        _command: Any, *_args: Any, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        mod, "build_workflow_run_command", _fake_build_workflow_run_command
    )
    monkeypatch.setattr(
        mod, "list_dispatched_run_ids", lambda *_args, **_kwargs: [100, 101]
    )
    monkeypatch.setattr(
        mod,
        "wait_for_new_dispatched_run_id",
        _fake_wait_for_new_dispatched_run_id,
    )
    monkeypatch.setattr(mod, "watch_run", _fake_watch_run)
    monkeypatch.setattr(mod.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))

    rc = _invoke_main(mod)
    assert rc != 0
    assert watch_calls["count"] == 0


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
    list_called = {"count": 0}
    wait_called = {"count": 0}
    watch_called = {"count": 0}
    subprocess_called = {"count": 0}

    def _fake_build_workflow_run_command(*_args: Any, **_kwargs: Any) -> list[str]:
        dispatch_called["count"] += 1
        return ["gh", "workflow", "run", "Experiment Archive Dry Run", "--ref", "main"]

    def _fake_list_dispatched_run_ids(*_args: Any, **_kwargs: Any) -> list[int]:
        list_called["count"] += 1
        return [100]

    def _fake_wait_for_new_dispatched_run_id(*_args: Any, **_kwargs: Any) -> int | None:
        wait_called["count"] += 1
        return 101

    def _fake_watch_run(_run_id: int) -> int:
        watch_called["count"] += 1
        return 0

    def _fake_subprocess_run(
        _command: Any, *_args: Any, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        subprocess_called["count"] += 1
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        mod, "build_workflow_run_command", _fake_build_workflow_run_command
    )
    monkeypatch.setattr(mod, "list_dispatched_run_ids", _fake_list_dispatched_run_ids)
    monkeypatch.setattr(
        mod,
        "wait_for_new_dispatched_run_id",
        _fake_wait_for_new_dispatched_run_id,
    )
    monkeypatch.setattr(mod, "watch_run", _fake_watch_run)
    monkeypatch.setattr(mod.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))

    rc = _invoke_main(mod)
    assert rc == 0
    assert dispatch_called["count"] == 1
    assert list_called["count"] == 0
    assert wait_called["count"] == 0
    assert watch_called["count"] == 0
    assert subprocess_called["count"] == 0
