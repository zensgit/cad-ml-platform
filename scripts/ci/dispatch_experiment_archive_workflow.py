#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from typing import Sequence

APPROVAL_PHRASE = "I_UNDERSTAND_DELETE_SOURCE"
MODE_DRY_RUN = "dry-run"
MODE_APPLY = "apply"
WORKFLOW_NAME_BY_MODE = {
    MODE_DRY_RUN: "Experiment Archive Dry Run",
    MODE_APPLY: "Experiment Archive Apply",
}


def resolve_workflow_name(mode: str) -> str:
    workflow_name = WORKFLOW_NAME_BY_MODE.get(mode)
    if workflow_name is None:
        raise ValueError(f"unsupported mode: {mode}")
    return workflow_name


def build_workflow_run_command(args: argparse.Namespace) -> list[str]:
    workflow_name = resolve_workflow_name(str(args.mode))
    command: list[str] = [
        "gh",
        "workflow",
        "run",
        workflow_name,
        "--ref",
        str(args.ref),
    ]

    shared_inputs = {
        "experiments_root": str(args.experiments_root),
        "archive_root": str(args.archive_root),
        "keep_latest_days": str(args.keep_latest_days),
        "today": str(args.today),
    }
    for key, value in shared_inputs.items():
        command.extend(["-f", f"{key}={value}"])

    if str(args.mode) == MODE_APPLY:
        apply_inputs = {
            "approval_phrase": str(args.approval_phrase),
            "dirs_csv": str(args.dirs_csv),
            "require_exists": str(args.require_exists),
        }
        for key, value in apply_inputs.items():
            command.extend(["-f", f"{key}={value}"])

    return command


def _build_find_latest_run_command(workflow_name: str, ref: str) -> list[str]:
    return [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow_name,
        "--branch",
        ref,
        "--event",
        "workflow_dispatch",
        "--json",
        "databaseId,createdAt",
        "--limit",
        "20",
    ]


def find_latest_dispatched_run_id(workflow_name: str, ref: str) -> int | None:
    command = _build_find_latest_run_command(workflow_name, ref)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, list) or not payload:
        return None

    latest = max(payload, key=lambda item: str(item.get("createdAt", "")))
    run_id = latest.get("databaseId")
    try:
        return int(run_id)
    except (TypeError, ValueError):
        return None


def watch_run(run_id: int) -> int:
    command = ["gh", "run", "watch", str(run_id), "--exit-status"]
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def _wait_for_new_run_id(
    workflow_name: str,
    ref: str,
    previous_run_id: int | None,
) -> int | None:
    for _ in range(20):
        run_id = find_latest_dispatched_run_id(workflow_name, ref)
        if run_id is None:
            time.sleep(3)
            continue

        if previous_run_id is None or run_id != previous_run_id:
            return run_id
        time.sleep(3)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch experiment archive workflows via gh CLI."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(MODE_DRY_RUN, MODE_APPLY),
        help="Choose workflow mode: dry-run or apply.",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="Git ref to run workflow against (default: main).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "After dispatch, watch latest workflow_dispatch run and return "
            "its exit code."
        ),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--experiments-root",
        default="reports/experiments",
        help="Workflow input: experiments_root.",
    )
    parser.add_argument(
        "--archive-root",
        default="",
        help="Workflow input: archive_root.",
    )
    parser.add_argument(
        "--keep-latest-days",
        type=int,
        default=7,
        help="Workflow input: keep_latest_days.",
    )
    parser.add_argument(
        "--today",
        default="",
        help="Workflow input: today (YYYYMMDD).",
    )

    parser.add_argument(
        "--approval-phrase",
        default="",
        help="Apply-only input: approval_phrase.",
    )
    parser.add_argument(
        "--dirs-csv",
        default="",
        help="Apply-only input: dirs_csv (comma-separated date dirs).",
    )
    parser.add_argument(
        "--require-exists",
        default="true",
        choices=("true", "false"),
        help="Apply-only input: require_exists (default: true).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if str(args.mode) == MODE_APPLY and str(args.approval_phrase) != APPROVAL_PHRASE:
        print(
            "error: --approval-phrase must be I_UNDERSTAND_DELETE_SOURCE "
            "when --mode=apply."
        )
        return 2

    workflow_name = resolve_workflow_name(str(args.mode))
    dispatch_command = build_workflow_run_command(args)
    if bool(args.print_only):
        print(shlex.join(dispatch_command))
        if bool(args.watch):
            print(
                shlex.join(_build_find_latest_run_command(workflow_name, str(args.ref)))
            )
            print("gh run watch <run_id> --exit-status")
        return 0

    previous_run_id: int | None = None
    if bool(args.watch):
        previous_run_id = find_latest_dispatched_run_id(workflow_name, str(args.ref))

    dispatch_result = subprocess.run(dispatch_command, check=False)
    if dispatch_result.returncode != 0:
        return int(dispatch_result.returncode)

    if not bool(args.watch):
        return 0

    run_id = _wait_for_new_run_id(workflow_name, str(args.ref), previous_run_id)
    if run_id is None:
        print(
            "error: failed to find latest workflow_dispatch run for "
            f"workflow={workflow_name!r} ref={args.ref!r}."
        )
        return 1

    return watch_run(run_id)


if __name__ == "__main__":
    raise SystemExit(main())
