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


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    main_func = module.main
    params = inspect.signature(main_func).parameters
    try:
        if len(params) == 0:
            old_argv = sys.argv
            try:
                sys.argv = ["watch_commit_workflows.py", *(argv or [])]
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


def test_split_csv_and_merge_items() -> None:
    from scripts.ci import watch_commit_workflows as mod

    items = mod._split_csv_items(" push,workflow_dispatch,, push ")
    merged = mod._merge_items(items, ["workflow_dispatch", "schedule"])
    assert items == ["push", "workflow_dispatch", "push"]
    assert merged == ["push", "workflow_dispatch", "schedule"]


def test_latest_runs_by_workflow_keeps_latest_database_id() -> None:
    from scripts.ci import watch_commit_workflows as mod

    runs = [
        mod.WorkflowRun(
            database_id=100,
            workflow_name="CI",
            status="completed",
            conclusion="success",
            url="u1",
            event="push",
        ),
        mod.WorkflowRun(
            database_id=120,
            workflow_name="CI",
            status="in_progress",
            conclusion="",
            url="u2",
            event="push",
        ),
        mod.WorkflowRun(
            database_id=110,
            workflow_name="Code Quality",
            status="completed",
            conclusion="success",
            url="u3",
            event="push",
        ),
    ]

    latest = mod.latest_runs_by_workflow(runs)
    assert [row.workflow_name for row in latest] == ["CI", "Code Quality"]
    assert [row.database_id for row in latest] == [120, 110]


def test_list_runs_for_sha_filters_by_sha_and_event(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    payload = [
        {
            "databaseId": 1,
            "headSha": "sha-a",
            "workflowName": "CI",
            "status": "completed",
            "conclusion": "success",
            "url": "u1",
            "event": "push",
        },
        {
            "databaseId": 2,
            "headSha": "sha-a",
            "workflowName": "CI Enhanced",
            "status": "completed",
            "conclusion": "success",
            "url": "u2",
            "event": "schedule",
        },
        {
            "databaseId": 3,
            "headSha": "sha-b",
            "workflowName": "CI",
            "status": "completed",
            "conclusion": "success",
            "url": "u3",
            "event": "push",
        },
    ]

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    runs = mod.list_runs_for_sha(head_sha="sha-a", events={"push"}, limit=100)
    assert len(runs) == 1
    assert runs[0].workflow_name == "CI"
    assert runs[0].database_id == 1


def test_list_runs_for_sha_raises_on_nonzero(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="boom",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    try:
        mod.list_runs_for_sha(head_sha="sha", events={"push"}, limit=100)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_main_print_only_does_not_execute_runtime_checks(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI",
            require_workflow=[],
            wait_timeout_seconds=30,
            poll_interval_seconds=5,
            list_limit=50,
            print_only=True,
        ),
    )

    called = {"check_gh_ready": 0, "resolve_head_sha": 0}

    def _fake_check_gh_ready() -> tuple[bool, str]:
        called["check_gh_ready"] += 1
        return True, ""

    def _fake_resolve_head_sha(_value: str) -> str:
        called["resolve_head_sha"] += 1
        return "sha"

    monkeypatch.setattr(mod, "check_gh_ready", _fake_check_gh_ready)
    monkeypatch.setattr(mod, "resolve_head_sha", _fake_resolve_head_sha)

    rc = _invoke_main(mod)
    assert rc == 0
    assert called["check_gh_ready"] == 0
    assert called["resolve_head_sha"] == 0


def test_main_success_when_required_workflows_complete(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI,Code Quality",
            require_workflow=[],
            wait_timeout_seconds=20,
            poll_interval_seconds=1,
            list_limit=100,
            print_only=False,
        ),
    )

    snapshots: list[list[mod.WorkflowRun]] = [
        [
            mod.WorkflowRun(
                database_id=101,
                workflow_name="CI",
                status="in_progress",
                conclusion="",
                url="u1",
                event="push",
            )
        ],
        [
            mod.WorkflowRun(
                database_id=101,
                workflow_name="CI",
                status="completed",
                conclusion="success",
                url="u1",
                event="push",
            ),
            mod.WorkflowRun(
                database_id=102,
                workflow_name="Code Quality",
                status="completed",
                conclusion="success",
                url="u2",
                event="push",
            ),
        ],
    ]

    def _fake_list_runs_for_sha(**_kwargs: Any) -> list[mod.WorkflowRun]:
        if snapshots:
            return snapshots.pop(0)
        return []

    clock = {"value": 0}

    def _fake_time() -> float:
        clock["value"] += 1
        return float(clock["value"])

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(mod, "list_runs_for_sha", _fake_list_runs_for_sha)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mod.time, "time", _fake_time)

    rc = _invoke_main(mod)
    assert rc == 0


def test_main_returns_non_zero_when_detecting_failure(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI",
            require_workflow=[],
            wait_timeout_seconds=20,
            poll_interval_seconds=1,
            list_limit=100,
            print_only=False,
        ),
    )

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(
        mod,
        "list_runs_for_sha",
        lambda **_kwargs: [
            mod.WorkflowRun(
                database_id=200,
                workflow_name="CI",
                status="completed",
                conclusion="failure",
                url="u1",
                event="push",
            )
        ],
    )

    rc = _invoke_main(mod)
    assert rc == 1


def test_main_timeout_when_required_workflow_missing(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI,Code Quality",
            require_workflow=[],
            wait_timeout_seconds=2,
            poll_interval_seconds=1,
            list_limit=100,
            missing_required_mode="wait",
            print_only=False,
        ),
    )

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(
        mod,
        "list_runs_for_sha",
        lambda **_kwargs: [
            mod.WorkflowRun(
                database_id=201,
                workflow_name="CI",
                status="completed",
                conclusion="success",
                url="u1",
                event="push",
            )
        ],
    )

    clock = {"value": 0}

    def _fake_time() -> float:
        clock["value"] += 1
        return float(clock["value"])

    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mod.time, "time", _fake_time)

    rc = _invoke_main(mod)
    assert rc == 1


def test_main_fail_fast_when_required_workflow_missing(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI,Code Quality",
            require_workflow=[],
            wait_timeout_seconds=120,
            poll_interval_seconds=1,
            list_limit=100,
            missing_required_mode="fail-fast",
            print_only=False,
        ),
    )

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(
        mod,
        "list_runs_for_sha",
        lambda **_kwargs: [
            mod.WorkflowRun(
                database_id=201,
                workflow_name="CI",
                status="completed",
                conclusion="success",
                url="u1",
                event="push",
            )
        ],
    )
    sleep_calls = {"count": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)

    rc = _invoke_main(mod)
    assert rc == 1
    assert sleep_calls["count"] == 0


def test_main_argument_validation_for_poll_interval(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="",
            require_workflow=[],
            wait_timeout_seconds=10,
            poll_interval_seconds=0,
            list_limit=100,
            print_only=False,
        ),
    )

    rc = _invoke_main(mod)
    assert rc == 2
