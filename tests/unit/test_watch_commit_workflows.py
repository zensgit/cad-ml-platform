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


def test_extract_auth_error_details_prefers_actionable_lines() -> None:
    from scripts.ci import watch_commit_workflows as mod

    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=(
            "github.com\n"
            "  X Failed to log in to github.com account zensgit (default)\n"
            "  - The token in default is invalid.\n"
            "  - To re-authenticate, run: gh auth login -h github.com\n"
        ),
        stderr="",
    )

    message = mod._extract_auth_error_details(result, "fallback")
    assert "Failed to log in" in message
    assert "token in default is invalid" in message
    assert "gh auth login -h github.com" in message


def test_check_gh_ready_reports_auth_context(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    responses = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="gh version", stderr=""),
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=(
                "github.com\n"
                "  X Failed to log in to github.com account zensgit (default)\n"
                "  - The token in default is invalid.\n"
                "  - To re-authenticate, run: gh auth login -h github.com\n"
            ),
            stderr="",
        ),
    ]

    def _fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if not responses:
            raise AssertionError("unexpected subprocess.run call")
        return responses.pop(0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    ok, reason = mod.check_gh_ready()
    assert ok is False
    assert "gh auth is not ready" in reason
    assert "token in default is invalid" in reason


def test_resolve_head_sha_expands_non_head_value(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    captured: dict[str, Any] = {}

    def _fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="530897b01c818a697ed46660f8b17bc8b1bb14c5\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    resolved = mod.resolve_head_sha("530897b")
    assert captured["args"] == ["git", "rev-parse", "530897b"]
    assert resolved == "530897b01c818a697ed46660f8b17bc8b1bb14c5"


def test_resolve_head_sha_uses_head_when_empty(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    captured: dict[str, Any] = {}

    def _fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="abc123\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    resolved = mod.resolve_head_sha("")
    assert captured["args"] == ["git", "rev-parse", "HEAD"]
    assert resolved == "abc123"


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


def test_main_failure_fail_fast_before_all_completed(monkeypatch: Any) -> None:
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
            failure_mode="fail-fast",
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
                database_id=301,
                workflow_name="CI",
                status="completed",
                conclusion="failure",
                url="u1",
                event="push",
            ),
            mod.WorkflowRun(
                database_id=302,
                workflow_name="Code Quality",
                status="in_progress",
                conclusion="",
                url="u2",
                event="push",
            ),
        ],
    )
    sleep_calls = {"count": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)

    rc = _invoke_main(mod)
    assert rc == 1
    assert sleep_calls["count"] == 0


def test_main_failure_wait_all_mode_waits_until_all_completed(monkeypatch: Any) -> None:
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
            failure_mode="wait-all",
            print_only=False,
        ),
    )

    snapshots: list[list[mod.WorkflowRun]] = [
        [
            mod.WorkflowRun(
                database_id=401,
                workflow_name="CI",
                status="completed",
                conclusion="failure",
                url="u1",
                event="push",
            ),
            mod.WorkflowRun(
                database_id=402,
                workflow_name="Code Quality",
                status="in_progress",
                conclusion="",
                url="u2",
                event="push",
            ),
        ],
        [
            mod.WorkflowRun(
                database_id=401,
                workflow_name="CI",
                status="completed",
                conclusion="failure",
                url="u1",
                event="push",
            ),
            mod.WorkflowRun(
                database_id=402,
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

    sleep_calls = {"count": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(mod, "list_runs_for_sha", _fake_list_runs_for_sha)
    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)

    rc = _invoke_main(mod)
    assert rc == 1
    assert sleep_calls["count"] >= 1


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


def test_main_argument_validation_for_heartbeat_interval(monkeypatch: Any) -> None:
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
            poll_interval_seconds=1,
            heartbeat_interval_seconds=-1,
            list_limit=100,
            print_only=False,
        ),
    )

    rc = _invoke_main(mod)
    assert rc == 2


def test_main_argument_validation_for_max_list_failures(monkeypatch: Any) -> None:
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
            poll_interval_seconds=1,
            heartbeat_interval_seconds=1,
            list_limit=100,
            max_list_failures=-1,
            print_only=False,
        ),
    )

    rc = _invoke_main(mod)
    assert rc == 2


def test_main_argument_validation_for_success_conclusions_empty(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="",
            require_workflow=[],
            success_conclusions_csv=" , ",
            wait_timeout_seconds=10,
            poll_interval_seconds=1,
            heartbeat_interval_seconds=1,
            list_limit=100,
            print_only=False,
        ),
    )

    rc = _invoke_main(mod)
    assert rc == 2


def test_main_allows_neutral_when_configured(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI",
            require_workflow=[],
            success_conclusions_csv="success,skipped,neutral",
            wait_timeout_seconds=20,
            poll_interval_seconds=1,
            heartbeat_interval_seconds=1,
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
                database_id=777,
                workflow_name="CI",
                status="completed",
                conclusion="neutral",
                url="u1",
                event="push",
            )
        ],
    )

    rc = _invoke_main(mod)
    assert rc == 0


def test_main_retries_run_list_failures_then_succeeds(monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI",
            require_workflow=[],
            wait_timeout_seconds=60,
            poll_interval_seconds=1,
            heartbeat_interval_seconds=5,
            list_limit=100,
            max_list_failures=1,
            print_only=False,
        ),
    )

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    attempts = {"count": 0}

    def _fake_list_runs_for_sha(**_kwargs: Any) -> list[mod.WorkflowRun]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("network temporary failure")
        return [
            mod.WorkflowRun(
                database_id=801,
                workflow_name="CI",
                status="completed",
                conclusion="success",
                url="u1",
                event="push",
            )
        ]

    sleep_calls = {"count": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(mod, "list_runs_for_sha", _fake_list_runs_for_sha)
    monkeypatch.setattr(mod.time, "sleep", _fake_sleep)

    rc = _invoke_main(mod)
    assert rc == 0
    assert attempts["count"] == 2
    assert sleep_calls["count"] >= 1


def test_main_fails_after_exceeding_max_list_failures(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from scripts.ci import watch_commit_workflows as mod

    summary_path = tmp_path / "watch-summary-failed.json"
    _patch_parsed_args(
        monkeypatch,
        _Args(
            sha="HEAD",
            events_csv="push",
            event=[],
            require_workflows_csv="CI",
            require_workflow=[],
            wait_timeout_seconds=60,
            poll_interval_seconds=1,
            heartbeat_interval_seconds=5,
            list_limit=100,
            max_list_failures=1,
            summary_json_out=str(summary_path),
            print_only=False,
        ),
    )

    monkeypatch.setattr(mod, "check_gh_ready", lambda: (True, ""))
    monkeypatch.setattr(mod, "resolve_head_sha", lambda _value: "sha")
    monkeypatch.setattr(
        mod,
        "list_runs_for_sha",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    rc = _invoke_main(mod)
    assert rc == 1
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "gh_run_list_failed"
    assert payload["consecutive_list_failures"] == 2
    assert payload["max_list_failures"] == 1


def test_main_print_only_writes_summary_json(tmp_path: Any, monkeypatch: Any) -> None:
    from scripts.ci import watch_commit_workflows as mod

    summary_path = tmp_path / "watch-summary.json"
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
            summary_json_out=str(summary_path),
            print_only=True,
        ),
    )

    rc = _invoke_main(mod)
    assert rc == 0
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["exit_code"] == 0
    assert payload["reason"] == "print_only"
    assert payload["success_conclusions"] == ["skipped", "success"]
    assert payload["max_list_failures"] == 3
    assert payload["consecutive_list_failures"] == 0
    assert payload["counts"]["observed"] == 0
    assert payload["events"] == ["push"]


def test_main_returns_non_zero_when_summary_json_write_fails(monkeypatch: Any) -> None:
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
            summary_json_out="/tmp/summary.json",
            print_only=True,
        ),
    )

    def _raise_write_error(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(mod, "_write_summary_json", _raise_write_error)

    rc = _invoke_main(mod)
    assert rc == 1
