#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Sequence


@dataclass(frozen=True)
class E2ECase:
    strict_value: str
    expected_conclusion: str


@dataclass(frozen=True)
class E2ERunResult:
    strict_value: str
    expected_conclusion: str
    run_id: int
    run_url: str
    conclusion: str
    dispatch_exit_code: int
    watch_exit_code: int
    matched_expectation: bool


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


def build_workflow_run_command(
    *,
    workflow: str,
    ref: str,
    review_pack_input_csv: str,
    strict_value: str,
) -> list[str]:
    return [
        "gh",
        "workflow",
        "run",
        str(workflow),
        "--ref",
        str(ref),
        "-f",
        f"review_pack_input_csv={review_pack_input_csv}",
        "-f",
        f"review_gate_strict={strict_value}",
    ]


def _build_list_dispatched_runs_command(
    workflow: str,
    ref: str,
    *,
    limit: int,
) -> list[str]:
    return [
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
        "databaseId",
        "--limit",
        str(max(1, int(limit))),
    ]


def list_dispatched_run_ids(workflow: str, ref: str, *, limit: int) -> list[int]:
    command = _build_list_dispatched_runs_command(workflow, ref, limit=limit)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    run_ids: list[int] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            run_ids.append(int(item.get("databaseId")))
        except (TypeError, ValueError):
            continue
    return run_ids


def wait_for_new_dispatched_run_id(
    *,
    workflow: str,
    ref: str,
    known_run_ids: Sequence[int],
    timeout_seconds: int,
    poll_interval_seconds: int,
    list_limit: int,
) -> int | None:
    known = set(int(item) for item in known_run_ids)
    deadline = time.time() + max(0, int(timeout_seconds))
    interval = max(1, int(poll_interval_seconds))
    while True:
        run_ids = list_dispatched_run_ids(workflow, ref, limit=list_limit)
        for run_id in run_ids:
            if run_id not in known:
                return int(run_id)
        if time.time() >= deadline:
            return None
        time.sleep(interval)


def watch_run(run_id: int) -> int:
    result = subprocess.run(["gh", "run", "watch", str(run_id), "--exit-status"], check=False)
    return int(result.returncode)


def get_run_conclusion_and_url(run_id: int) -> tuple[str, str]:
    result = subprocess.run(
        ["gh", "run", "view", str(run_id), "--json", "conclusion,url"],
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
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[str, str]:
    deadline = time.time() + max(0, int(timeout_seconds))
    interval = max(1, int(poll_interval_seconds))
    last_url = ""
    while True:
        conclusion, run_url = get_run_conclusion_and_url(int(run_id))
        if run_url:
            last_url = run_url
        if conclusion:
            return (conclusion, run_url or last_url)
        if time.time() >= deadline:
            return ("", run_url or last_url)
        time.sleep(interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch evaluation-report workflow twice (strict=false then strict=true) "
            "and assert expected conclusions."
        )
    )
    parser.add_argument("--workflow", default="evaluation-report.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument(
        "--review-pack-input-csv",
        default="tests/fixtures/ci/graph2d_review_pack_input.csv",
        help="Input CSV path passed to workflow_dispatch.review_pack_input_csv",
    )
    parser.add_argument("--wait-timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-interval-seconds", type=int, default=3)
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without dispatching workflows.",
    )
    return parser


def _build_cases() -> list[E2ECase]:
    return [
        E2ECase(strict_value="false", expected_conclusion="success"),
        E2ECase(strict_value="true", expected_conclusion="failure"),
    ]


def _serialize_results(
    *,
    workflow: str,
    ref: str,
    review_pack_input_csv: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
    list_limit: int,
    results: Sequence[E2ERunResult],
    overall_exit_code: int,
) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "ref": ref,
        "review_pack_input_csv": review_pack_input_csv,
        "wait_timeout_seconds": timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "list_limit": list_limit,
        "overall_exit_code": int(overall_exit_code),
        "runs": [
            {
                "strict_value": item.strict_value,
                "expected_conclusion": item.expected_conclusion,
                "run_id": int(item.run_id),
                "run_url": item.run_url,
                "conclusion": item.conclusion,
                "dispatch_exit_code": int(item.dispatch_exit_code),
                "watch_exit_code": int(item.watch_exit_code),
                "matched_expectation": bool(item.matched_expectation),
            }
            for item in results
        ],
    }


def _write_output_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cases = _build_cases()
    commands = [
        build_workflow_run_command(
            workflow=str(args.workflow),
            ref=str(args.ref),
            review_pack_input_csv=str(args.review_pack_input_csv),
            strict_value=case.strict_value,
        )
        for case in cases
    ]

    if bool(args.print_only):
        for command in commands:
            print(shlex.join(command))
        print(
            shlex.join(
                _build_list_dispatched_runs_command(
                    str(args.workflow), str(args.ref), limit=int(args.list_limit)
                )
            )
        )
        print("gh run watch <run_id> --exit-status")
        return 0

    is_ready, reason = check_gh_ready()
    if not is_ready:
        print(f"error: {reason}")
        return 1

    all_results: list[E2ERunResult] = []
    for case in cases:
        known_ids = list_dispatched_run_ids(
            str(args.workflow),
            str(args.ref),
            limit=int(args.list_limit),
        )
        dispatch_cmd = build_workflow_run_command(
            workflow=str(args.workflow),
            ref=str(args.ref),
            review_pack_input_csv=str(args.review_pack_input_csv),
            strict_value=case.strict_value,
        )
        print(f"Dispatching strict={case.strict_value}: {shlex.join(dispatch_cmd)}")
        dispatch_result = subprocess.run(dispatch_cmd, check=False)
        dispatch_exit_code = int(dispatch_result.returncode)
        if dispatch_exit_code != 0:
            print(f"error: dispatch failed for strict={case.strict_value}")
            return dispatch_exit_code

        run_id = wait_for_new_dispatched_run_id(
            workflow=str(args.workflow),
            ref=str(args.ref),
            known_run_ids=known_ids,
            timeout_seconds=int(args.wait_timeout_seconds),
            poll_interval_seconds=int(args.poll_interval_seconds),
            list_limit=int(args.list_limit),
        )
        if run_id is None:
            print(
                "error: timed out waiting for dispatched run id "
                f"(strict={case.strict_value}, workflow={args.workflow}, ref={args.ref})"
            )
            return 1

        print(f"Watching run {run_id} (strict={case.strict_value})")
        watch_exit_code = watch_run(int(run_id))
        conclusion, run_url = wait_for_run_conclusion(
            run_id=int(run_id),
            timeout_seconds=int(args.wait_timeout_seconds),
            poll_interval_seconds=int(args.poll_interval_seconds),
        )
        matched_expectation = str(conclusion) == case.expected_conclusion

        all_results.append(
            E2ERunResult(
                strict_value=case.strict_value,
                expected_conclusion=case.expected_conclusion,
                run_id=int(run_id),
                run_url=run_url,
                conclusion=str(conclusion),
                dispatch_exit_code=dispatch_exit_code,
                watch_exit_code=watch_exit_code,
                matched_expectation=matched_expectation,
            )
        )
        print(
            "Result: "
            f"strict={case.strict_value} expected={case.expected_conclusion} "
            f"actual={conclusion or 'unknown'} watch_exit={watch_exit_code} "
            f"url={run_url}"
        )

    overall_exit_code = 0 if all(item.matched_expectation for item in all_results) else 1
    payload = _serialize_results(
        workflow=str(args.workflow),
        ref=str(args.ref),
        review_pack_input_csv=str(args.review_pack_input_csv),
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
        list_limit=int(args.list_limit),
        results=all_results,
        overall_exit_code=overall_exit_code,
    )
    if str(args.output_json).strip():
        _write_output_json(str(args.output_json), payload)
        print(f"Wrote summary: {args.output_json}")

    if overall_exit_code == 0:
        print("Strict gate e2e validation passed.")
    else:
        print("Strict gate e2e validation failed.")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
