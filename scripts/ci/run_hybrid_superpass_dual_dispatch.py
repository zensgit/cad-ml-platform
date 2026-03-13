#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
from typing import Any, Optional, Sequence


def _normalize_optional(value: str) -> str:
    return str(value or "").strip()


def _write_output_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _read_output_json(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _build_dispatch_command(
    *,
    dispatch_script: str,
    workflow: str,
    ref: str,
    repo: str,
    wait_timeout_seconds: int,
    poll_interval_seconds: int,
    list_limit: int,
    output_json: str,
    expected_conclusion: str,
    missing_mode: str,
    fail_on_failed: str,
    dispatch_trace_id: str,
) -> list[str]:
    command = [
        sys.executable,
        str(dispatch_script),
        "--workflow",
        str(workflow),
        "--ref",
        str(ref),
        "--hybrid-superpass-enable",
        "true",
        "--hybrid-superpass-missing-mode",
        str(missing_mode),
        "--hybrid-superpass-fail-on-failed",
        str(fail_on_failed),
        "--expected-conclusion",
        str(expected_conclusion),
        "--wait-timeout-seconds",
        str(int(wait_timeout_seconds)),
        "--poll-interval-seconds",
        str(int(poll_interval_seconds)),
        "--list-limit",
        str(int(list_limit)),
        "--output-json",
        str(output_json),
        "--dispatch-trace-id",
        str(dispatch_trace_id),
    ]
    repo_value = _normalize_optional(repo)
    if repo_value:
        command.extend(["--repo", repo_value])
    return command


def _build_compare_command(
    *,
    compare_script: str,
    fail_output_json: str,
    success_output_json: str,
    compare_output_json: str,
    compare_output_md: str,
    strict: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(compare_script),
        "--fail-json",
        str(fail_output_json),
        "--success-json",
        str(success_output_json),
        "--output-json",
        str(compare_output_json),
        "--output-md",
        str(compare_output_md),
    ]
    if strict:
        command.append("--strict")
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fail/success hybrid superpass dispatch in parallel and compare reports."
        )
    )
    parser.add_argument("--workflow", default="hybrid-superpass-e2e.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--repo", default="", help="Optional owner/repo for gh commands.")
    parser.add_argument("--wait-timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=int, default=3)
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--fail-output-json", required=True)
    parser.add_argument("--success-output-json", required=True)
    parser.add_argument("--compare-output-json", required=True)
    parser.add_argument("--compare-output-md", required=True)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional orchestrator summary JSON path.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--dispatch-script",
        default="scripts/ci/dispatch_hybrid_superpass_workflow.py",
    )
    parser.add_argument(
        "--compare-script",
        default="scripts/ci/compare_hybrid_superpass_reports.py",
    )
    parser.add_argument(
        "--dispatch-trace-prefix",
        default="",
        help="Trace prefix for fail/success dispatch trace id.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    trace_prefix = _normalize_optional(str(args.dispatch_trace_prefix))
    if not trace_prefix:
        trace_prefix = f"dsp-{uuid.uuid4().hex[:12]}"

    fail_trace_id = f"{trace_prefix}-fail"
    success_trace_id = f"{trace_prefix}-success"

    fail_dispatch_cmd = _build_dispatch_command(
        dispatch_script=str(args.dispatch_script),
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        wait_timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
        list_limit=int(args.list_limit),
        output_json=str(args.fail_output_json),
        expected_conclusion="failure",
        missing_mode="fail",
        fail_on_failed="true",
        dispatch_trace_id=fail_trace_id,
    )
    success_dispatch_cmd = _build_dispatch_command(
        dispatch_script=str(args.dispatch_script),
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        wait_timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
        list_limit=int(args.list_limit),
        output_json=str(args.success_output_json),
        expected_conclusion="success",
        missing_mode="skip",
        fail_on_failed="false",
        dispatch_trace_id=success_trace_id,
    )
    compare_cmd = _build_compare_command(
        compare_script=str(args.compare_script),
        fail_output_json=str(args.fail_output_json),
        success_output_json=str(args.success_output_json),
        compare_output_json=str(args.compare_output_json),
        compare_output_md=str(args.compare_output_md),
        strict=bool(args.strict),
    )

    print("fail_dispatch_command=" + shlex.join(fail_dispatch_cmd))
    print("success_dispatch_command=" + shlex.join(success_dispatch_cmd))
    print("compare_command=" + shlex.join(compare_cmd))

    payload: dict[str, Any] = {
        "mode": "run",
        "dispatch_trace_prefix": trace_prefix,
        "fail_dispatch_trace_id": fail_trace_id,
        "success_dispatch_trace_id": success_trace_id,
        "fail_dispatch_command": fail_dispatch_cmd,
        "success_dispatch_command": success_dispatch_cmd,
        "compare_command": compare_cmd,
        "fail_dispatch_exit_code": None,
        "success_dispatch_exit_code": None,
        "compare_exit_code": None,
        "overall_exit_code": 1,
    }

    if args.print_only:
        payload["mode"] = "print_only"
        payload["overall_exit_code"] = 0
        _write_output_json(str(args.compare_output_json), payload)
        if _normalize_optional(str(args.output_json)):
            _write_output_json(str(args.output_json), payload)
        return 0

    fail_process = subprocess.Popen(fail_dispatch_cmd)
    success_process = subprocess.Popen(success_dispatch_cmd)

    fail_exit_code = int(fail_process.wait())
    success_exit_code = int(success_process.wait())

    compare_result = subprocess.run(compare_cmd, check=False)
    compare_exit_code = int(compare_result.returncode)

    compare_payload = _read_output_json(str(args.compare_output_json))
    payload = {
        **compare_payload,
        **payload,
        "mode": "run",
        "fail_dispatch_exit_code": fail_exit_code,
        "success_dispatch_exit_code": success_exit_code,
        "compare_exit_code": compare_exit_code,
    }

    overall_exit_code = (
        0
        if fail_exit_code == 0 and success_exit_code == 0 and compare_exit_code == 0
        else 1
    )
    payload["overall_exit_code"] = overall_exit_code
    _write_output_json(str(args.compare_output_json), payload)
    if _normalize_optional(str(args.output_json)):
        _write_output_json(str(args.output_json), payload)

    return int(overall_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
