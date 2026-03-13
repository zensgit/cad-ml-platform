#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid
from typing import Any, Optional, Sequence


def _extract_short_error(
    result: subprocess.CompletedProcess[str],
    fallback: str,
) -> str:
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        return fallback
    return output.splitlines()[0]


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


def _normalize_optional(value: str) -> str:
    return str(value or "").strip()


def build_workflow_run_command(
    *,
    workflow: str,
    ref: str,
    repo: str,
    target_repo: str,
    target_ref: str,
    target_workflow: str,
    dispatch_trace_id: str = "",
    dual_wait_timeout_seconds: str = "900",
    dual_poll_interval_seconds: str = "3",
    dual_list_limit: str = "20",
    strict_require_distinct_run_ids: str = "true",
    strict_require_trace_pair: str = "true",
) -> list[str]:
    command = [
        "gh",
        "workflow",
        "run",
        str(workflow),
        "--ref",
        str(ref),
    ]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])

    def _append_if_present(key: str, raw: str) -> None:
        value = _normalize_optional(raw)
        if value:
            command.extend(["-f", f"{key}={value}"])

    _append_if_present("target_repo", target_repo)
    _append_if_present("target_ref", target_ref)
    _append_if_present("target_workflow", target_workflow)
    _append_if_present("dispatch_trace_id", dispatch_trace_id)
    _append_if_present("dual_wait_timeout_seconds", dual_wait_timeout_seconds)
    _append_if_present("dual_poll_interval_seconds", dual_poll_interval_seconds)
    _append_if_present("dual_list_limit", dual_list_limit)
    _append_if_present(
        "strict_require_distinct_run_ids",
        strict_require_distinct_run_ids,
    )
    _append_if_present("strict_require_trace_pair", strict_require_trace_pair)
    return command


def _build_list_dispatched_runs_command(
    workflow: str,
    ref: str,
    repo: str,
    *,
    limit: int,
) -> list[str]:
    command = [
        "gh",
        "run",
        "list",
        "--workflow",
        str(workflow),
        "--branch",
        str(ref),
        "--event",
        "workflow_dispatch",
        "--json",
        "databaseId,displayTitle",
        "--limit",
        str(max(1, int(limit))),
    ]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])
    return command


def list_dispatched_runs(
    workflow: str,
    ref: str,
    repo: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    command = _build_list_dispatched_runs_command(workflow, ref, repo, limit=limit)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    runs: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            run_id = int(item.get("databaseId"))
        except (TypeError, ValueError):
            continue
        runs.append(
            {
                "databaseId": run_id,
                "displayTitle": str(item.get("displayTitle") or ""),
            }
        )
    return runs


def list_dispatched_run_ids(
    workflow: str,
    ref: str,
    repo: str,
    *,
    limit: int,
) -> list[int]:
    runs = list_dispatched_runs(workflow, ref, repo, limit=limit)
    return [int(item["databaseId"]) for item in runs if "databaseId" in item]


def wait_for_new_dispatched_run_id(
    *,
    workflow: str,
    ref: str,
    repo: str,
    known_run_ids: Sequence[int],
    timeout_seconds: int,
    poll_interval_seconds: int,
    list_limit: int,
    dispatch_trace_id: str = "",
) -> Optional[int]:
    known = set(int(item) for item in known_run_ids)
    deadline = time.time() + max(0, int(timeout_seconds))
    interval = max(1, int(poll_interval_seconds))
    trace_token = _normalize_optional(dispatch_trace_id)

    while True:
        runs = list_dispatched_runs(workflow, ref, repo, limit=list_limit)
        for run in runs:
            run_id = int(run.get("databaseId", 0))
            if run_id <= 0:
                continue
            if trace_token and trace_token not in str(run.get("displayTitle") or ""):
                continue
            if run_id not in known:
                return int(run_id)

        if time.time() >= deadline:
            return None
        time.sleep(interval)


def watch_run(run_id: int, repo: str) -> int:
    command = ["gh", "run", "watch", str(run_id), "--exit-status"]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def get_run_conclusion_and_url(run_id: int, repo: str) -> tuple[str, str]:
    command = ["gh", "run", "view", str(run_id), "--json", "conclusion,url"]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ("", "")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ("", "")
    if not isinstance(payload, dict):
        return ("", "")
    return (str(payload.get("conclusion") or ""), str(payload.get("url") or ""))


def wait_for_run_conclusion(
    *,
    run_id: int,
    repo: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[str, str]:
    deadline = time.time() + max(0, int(timeout_seconds))
    interval = max(1, int(poll_interval_seconds))
    last_url = ""
    while True:
        conclusion, run_url = get_run_conclusion_and_url(int(run_id), str(repo))
        if run_url:
            last_url = run_url
        if conclusion:
            return (conclusion, run_url or last_url)
        if time.time() >= deadline:
            return ("", run_url or last_url)
        time.sleep(interval)


def _write_output_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch hybrid-superpass-nightly workflow and optionally watch result."
    )
    parser.add_argument("--workflow", default="hybrid-superpass-nightly.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--repo", default="", help="Optional owner/repo for gh commands.")
    parser.add_argument("--target-repo", default="")
    parser.add_argument("--target-ref", default="main")
    parser.add_argument("--target-workflow", default="hybrid-superpass-e2e.yml")
    parser.add_argument("--dual-wait-timeout-seconds", default="900")
    parser.add_argument("--dual-poll-interval-seconds", default="3")
    parser.add_argument("--dual-list-limit", default="20")
    parser.add_argument("--strict-require-distinct-run-ids", default="true")
    parser.add_argument("--strict-require-trace-pair", default="true")
    parser.add_argument(
        "--dispatch-trace-id",
        default="",
        help="Optional trace id for run correlation under concurrent dispatch.",
    )
    parser.add_argument(
        "--expected-conclusion",
        default="success",
        choices=("success", "failure", "cancelled"),
        help="Expected workflow run conclusion.",
    )
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=3)
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without dispatching workflows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    resolved_dispatch_trace_id = _normalize_optional(str(args.dispatch_trace_id))
    if (
        not resolved_dispatch_trace_id
        and str(args.workflow).strip() == "hybrid-superpass-nightly.yml"
    ):
        resolved_dispatch_trace_id = f"nsp-{uuid.uuid4().hex[:12]}"

    dispatch_cmd = build_workflow_run_command(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        target_repo=str(args.target_repo),
        target_ref=str(args.target_ref),
        target_workflow=str(args.target_workflow),
        dispatch_trace_id=resolved_dispatch_trace_id,
        dual_wait_timeout_seconds=str(args.dual_wait_timeout_seconds),
        dual_poll_interval_seconds=str(args.dual_poll_interval_seconds),
        dual_list_limit=str(args.dual_list_limit),
        strict_require_distinct_run_ids=str(args.strict_require_distinct_run_ids),
        strict_require_trace_pair=str(args.strict_require_trace_pair),
    )

    watch_hint_cmd = ["gh", "run", "watch", "<run_id>", "--exit-status"]
    view_hint_cmd = ["gh", "run", "view", "<run_id>", "--json", "conclusion,url"]
    if str(args.repo).strip():
        watch_hint_cmd.extend(["--repo", str(args.repo)])
        view_hint_cmd.extend(["--repo", str(args.repo)])
    watch_hint = shlex.join(watch_hint_cmd)
    view_hint = shlex.join(view_hint_cmd)

    print("dispatch_command=" + shlex.join(dispatch_cmd))
    if resolved_dispatch_trace_id:
        print("dispatch_trace_id=" + resolved_dispatch_trace_id)
    print("watch_hint=" + watch_hint)
    print("view_hint=" + view_hint)

    payload: dict[str, Any] = {
        "dispatch_command": dispatch_cmd,
        "dispatch_trace_id": resolved_dispatch_trace_id,
        "run_id": None,
        "run_url": "",
        "conclusion": "",
        "expected_conclusion": str(args.expected_conclusion),
        "matched_expectation": False,
        "overall_exit_code": 1,
    }

    if args.print_only:
        payload.update({"mode": "print_only", "overall_exit_code": 0})
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 0

    ready, message = check_gh_ready()
    if not ready:
        print(message)
        payload.update({"reason": message})
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 1

    known_runs = list_dispatched_run_ids(
        str(args.workflow),
        str(args.ref),
        str(args.repo),
        limit=int(args.list_limit),
    )

    dispatch_result = subprocess.run(dispatch_cmd, capture_output=True, text=True, check=False)
    if dispatch_result.returncode != 0:
        msg = _extract_short_error(dispatch_result, "failed to dispatch workflow")
        print(f"error: {msg}")
        payload.update(
            {
                "reason": "dispatch_failed",
                "dispatch_exit_code": int(dispatch_result.returncode),
                "dispatch_stdout": dispatch_result.stdout or "",
                "dispatch_stderr": dispatch_result.stderr or "",
            }
        )
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 1

    run_id = wait_for_new_dispatched_run_id(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        known_run_ids=known_runs,
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
        list_limit=int(args.list_limit),
        dispatch_trace_id=resolved_dispatch_trace_id,
    )
    if run_id is None:
        print("error: timed out waiting for dispatched workflow run id.")
        payload.update({"reason": "run_id_timeout"})
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 1

    watch_exit_code = watch_run(int(run_id), str(args.repo))
    conclusion, run_url = wait_for_run_conclusion(
        run_id=int(run_id),
        repo=str(args.repo),
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
    )

    matched = conclusion == str(args.expected_conclusion)
    overall_exit_code = 0 if matched else 1

    payload.update(
        {
            "run_id": int(run_id),
            "run_url": run_url,
            "conclusion": conclusion,
            "matched_expectation": bool(matched),
            "watch_exit_code": int(watch_exit_code),
            "overall_exit_code": int(overall_exit_code),
        }
    )
    if args.output_json:
        _write_output_json(str(args.output_json), payload)

    summary = (
        "result run_id={} conclusion={} expected={} "
        "matched_expectation={} watch_exit_code={}"
    )
    print(
        summary.format(
            run_id,
            conclusion,
            args.expected_conclusion,
            matched,
            watch_exit_code,
        )
    )
    if run_url:
        print(f"run_url={run_url}")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
