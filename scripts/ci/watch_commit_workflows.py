#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

SUCCESS_CONCLUSIONS = {"success", "skipped"}


@dataclass(frozen=True)
class WorkflowRun:
    database_id: int
    workflow_name: str
    status: str
    conclusion: str
    url: str
    event: str


def _log(message: str) -> None:
    print(message, flush=True)


def _extract_short_error(
    result: subprocess.CompletedProcess[str], fallback: str
) -> str:
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        return fallback
    return output.splitlines()[0]


def _split_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _merge_items(*groups: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            normalized = str(item).strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
    return merged


def _int_with_default(value: object, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def check_gh_ready() -> tuple[bool, str]:
    version_result = subprocess.run(
        ["gh", "--version"], capture_output=True, text=True, check=False
    )
    if version_result.returncode != 0:
        return (
            False,
            "gh is not available: "
            f"{_extract_short_error(version_result, 'failed to run gh --version')}",
        )

    auth_result = subprocess.run(
        ["gh", "auth", "status"], capture_output=True, text=True, check=False
    )
    if auth_result.returncode != 0:
        return (
            False,
            "gh auth is not ready: "
            f"{_extract_short_error(auth_result, 'failed to run gh auth status')}",
        )
    return True, ""


def resolve_head_sha(value: str) -> str:
    normalized = value.strip()
    if normalized and normalized.upper() != "HEAD":
        return normalized

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        message = _extract_short_error(result, "failed to run git rev-parse HEAD")
        raise RuntimeError(f"failed to resolve HEAD sha: {message}")

    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("failed to resolve HEAD sha: empty output")
    return sha


def _build_list_runs_command(limit: int) -> list[str]:
    return [
        "gh",
        "run",
        "list",
        "--json",
        "databaseId,headSha,workflowName,status,conclusion,url,event",
        "--limit",
        str(max(1, int(limit))),
    ]


def list_runs_for_sha(
    *,
    head_sha: str,
    events: set[str],
    limit: int,
) -> list[WorkflowRun]:
    command = _build_list_runs_command(limit)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = _extract_short_error(result, "failed to list gh runs")
        raise RuntimeError(message)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to decode gh run list json: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("invalid gh run list payload: expected list")

    runs: list[WorkflowRun] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("headSha") or "") != head_sha:
            continue

        event = str(item.get("event") or "")
        if events and event not in events:
            continue

        workflow_name = str(item.get("workflowName") or "").strip()
        if not workflow_name:
            continue

        try:
            run_id = int(item.get("databaseId"))
        except (TypeError, ValueError):
            continue

        runs.append(
            WorkflowRun(
                database_id=run_id,
                workflow_name=workflow_name,
                status=str(item.get("status") or "").strip(),
                conclusion=str(item.get("conclusion") or "").strip(),
                url=str(item.get("url") or "").strip(),
                event=event,
            )
        )
    return runs


def latest_runs_by_workflow(runs: Sequence[WorkflowRun]) -> list[WorkflowRun]:
    latest: dict[str, WorkflowRun] = {}
    for run in sorted(runs, key=lambda row: row.database_id, reverse=True):
        if run.workflow_name not in latest:
            latest[run.workflow_name] = run
    return sorted(latest.values(), key=lambda row: row.workflow_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait until workflows for a commit SHA reach terminal states."
    )
    parser.add_argument(
        "--sha",
        default="HEAD",
        help="Commit SHA to watch (default: HEAD).",
    )
    parser.add_argument(
        "--events-csv",
        default="push",
        help="Comma-separated events to include (default: push).",
    )
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        help="Additional allowed event (repeatable).",
    )
    parser.add_argument(
        "--require-workflows-csv",
        default="",
        help="Comma-separated required workflow names.",
    )
    parser.add_argument(
        "--require-workflow",
        action="append",
        default=[],
        help="Required workflow name (repeatable).",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=1800,
        help="Max wait seconds before timeout (default: 1800).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=20,
        help="Polling interval in seconds (default: 20).",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=120,
        help=(
            "Print heartbeat when state is unchanged for N seconds. "
            "Set 0 to disable (default: 120)."
        ),
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=100,
        help="gh run list limit (default: 100).",
    )
    parser.add_argument(
        "--missing-required-mode",
        choices=("fail-fast", "wait"),
        default="fail-fast",
        help=(
            "Behavior when required workflows are missing and all observed runs "
            "are completed (default: fail-fast)."
        ),
    )
    parser.add_argument(
        "--failure-mode",
        choices=("fail-fast", "wait-all"),
        default="fail-fast",
        help=(
            "Behavior when non-success conclusions are detected. "
            "fail-fast returns immediately; wait-all waits for all workflows."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the gh command and exit without execution.",
    )
    return parser


def _print_snapshot(runs: Sequence[WorkflowRun], missing_required: Sequence[str]) -> None:
    total = len(runs)
    completed = sum(1 for run in runs if run.status == "completed")
    failed = sum(
        1
        for run in runs
        if run.status == "completed" and run.conclusion not in SUCCESS_CONCLUSIONS
    )
    _log(
        "status: "
        f"observed={total} completed={completed} failed={failed} "
        f"missing_required={len(missing_required)}"
    )
    for run in runs:
        conclusion = run.conclusion or "-"
        _log(f" - {run.workflow_name}: {run.status}/{conclusion} ({run.database_id})")
    if missing_required:
        _log(f" missing required workflows: {', '.join(missing_required)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    wait_timeout_seconds = _int_with_default(args.wait_timeout_seconds, 1800)
    poll_interval_seconds = _int_with_default(args.poll_interval_seconds, 20)
    heartbeat_interval_seconds = _int_with_default(args.heartbeat_interval_seconds, 120)
    list_limit = _int_with_default(args.list_limit, 100)

    if wait_timeout_seconds < 0:
        _log("error: --wait-timeout-seconds must be >= 0")
        return 2
    if poll_interval_seconds <= 0:
        _log("error: --poll-interval-seconds must be > 0")
        return 2
    if heartbeat_interval_seconds < 0:
        _log("error: --heartbeat-interval-seconds must be >= 0")
        return 2
    if list_limit <= 0:
        _log("error: --list-limit must be > 0")
        return 2

    events = set(_merge_items(_split_csv_items(str(args.events_csv)), list(args.event)))
    required_workflows = _merge_items(
        _split_csv_items(str(args.require_workflows_csv)),
        list(args.require_workflow),
    )

    command_preview = _build_list_runs_command(list_limit)
    missing_required_mode = str(args.missing_required_mode or "fail-fast")
    failure_mode = str(args.failure_mode or "fail-fast")

    if bool(args.print_only):
        _log(shlex.join(command_preview))
        _log(f"# events={sorted(events)}")
        _log(f"# required_workflows={required_workflows}")
        _log(f"# missing_required_mode={missing_required_mode}")
        _log(f"# failure_mode={failure_mode}")
        _log(f"# heartbeat_interval_seconds={heartbeat_interval_seconds}")
        return 0

    is_ready, reason = check_gh_ready()
    if not is_ready:
        _log(f"error: {reason}")
        return 1

    try:
        head_sha = resolve_head_sha(str(args.sha))
    except RuntimeError as exc:
        _log(f"error: {exc}")
        return 1

    deadline = time.time() + wait_timeout_seconds
    last_snapshot_key: tuple[object, ...] | None = None
    last_progress_log_at = time.time()

    while True:
        try:
            all_runs = list_runs_for_sha(
                head_sha=head_sha,
                events=events,
                limit=list_limit,
            )
        except RuntimeError as exc:
            _log(f"error: {exc}")
            return 1

        runs = latest_runs_by_workflow(all_runs)
        workflow_names = {run.workflow_name for run in runs}
        missing_required = [
            name for name in required_workflows if name not in workflow_names
        ]

        snapshot_key = (
            tuple((run.workflow_name, run.status, run.conclusion) for run in runs),
            tuple(missing_required),
        )
        if snapshot_key != last_snapshot_key:
            _print_snapshot(runs, missing_required)
            last_snapshot_key = snapshot_key
            last_progress_log_at = time.time()
        elif heartbeat_interval_seconds > 0:
            now = time.time()
            if now - last_progress_log_at >= heartbeat_interval_seconds:
                _log(
                    "heartbeat: workflows unchanged, still waiting "
                    f"(observed={len(runs)})."
                )
                last_progress_log_at = now

        has_runs = bool(runs)
        all_completed = has_runs and all(run.status == "completed" for run in runs)
        have_required = not missing_required

        failed = [
            run
            for run in runs
            if run.status == "completed" and run.conclusion not in SUCCESS_CONCLUSIONS
        ]
        if failed and failure_mode == "fail-fast":
            _log("error: detected non-success workflow conclusions.")
            return 1

        if all_completed and missing_required and missing_required_mode == "fail-fast":
            _log("error: required workflows are missing after observed runs completed.")
            return 1

        if all_completed and have_required:
            if failed:
                _log("error: detected non-success workflow conclusions.")
                return 1
            _log("all observed workflows completed successfully.")
            return 0

        if time.time() >= deadline:
            if not has_runs:
                _log("error: timeout while waiting for matching workflows to appear.")
            elif missing_required:
                _log("error: timeout with missing required workflows.")
            else:
                _log("error: timeout while waiting for workflows to complete.")
            return 1
        time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
