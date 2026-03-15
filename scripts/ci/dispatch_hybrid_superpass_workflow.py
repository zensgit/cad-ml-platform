#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import time
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
    hybrid_superpass_enable: str,
    hybrid_superpass_missing_mode: str,
    hybrid_superpass_fail_on_failed: str,
    hybrid_blind_enable: str = "",
    hybrid_blind_dxf_dir: str = "",
    hybrid_blind_fail_on_gate_failed: str = "",
    hybrid_blind_strict_require_real_data: str = "",
    hybrid_calibration_enable: str = "",
    hybrid_calibration_input_csv: str = "",
) -> list[str]:
    command = [
        "gh",
        "workflow",
        "run",
        str(workflow),
        "--ref",
        str(ref),
        "-f",
        f"hybrid_superpass_enable={hybrid_superpass_enable}",
        "-f",
        f"hybrid_superpass_missing_mode={hybrid_superpass_missing_mode}",
        "-f",
        f"hybrid_superpass_fail_on_failed={hybrid_superpass_fail_on_failed}",
    ]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])

    def _append_if_present(key: str, raw: str) -> None:
        value = _normalize_optional(raw)
        if value:
            command.extend(["-f", f"{key}={value}"])

    _append_if_present("hybrid_blind_enable", hybrid_blind_enable)
    _append_if_present("hybrid_blind_dxf_dir", hybrid_blind_dxf_dir)
    _append_if_present(
        "hybrid_blind_fail_on_gate_failed", hybrid_blind_fail_on_gate_failed
    )
    _append_if_present(
        "hybrid_blind_strict_require_real_data",
        hybrid_blind_strict_require_real_data,
    )
    _append_if_present("hybrid_calibration_enable", hybrid_calibration_enable)
    _append_if_present("hybrid_calibration_input_csv", hybrid_calibration_input_csv)
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
        "databaseId",
        "--limit",
        str(max(1, int(limit))),
    ]
    if _normalize_optional(repo):
        command.extend(["--repo", _normalize_optional(repo)])
    return command


def list_dispatched_run_ids(
    workflow: str, ref: str, repo: str, *, limit: int
) -> list[int]:
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
    repo: str,
    known_run_ids: Sequence[int],
    timeout_seconds: int,
    poll_interval_seconds: int,
    list_limit: int,
) -> Optional[int]:
    known = set(int(item) for item in known_run_ids)
    deadline = time.time() + max(0, int(timeout_seconds))
    interval = max(1, int(poll_interval_seconds))
    while True:
        run_ids = list_dispatched_run_ids(workflow, ref, repo, limit=list_limit)
        for run_id in run_ids:
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
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
    )


def fetch_remote_workflow_text(repo: str, workflow: str, ref: str) -> str:
    repo_text = _normalize_optional(repo)
    if not repo_text:
        return ""
    endpoint = f"repos/{repo_text}/contents/.github/workflows/{workflow}"
    result = subprocess.run(
        ["gh", "api", f"{endpoint}?ref={ref}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    encoded = str(payload.get("content") or "").replace("\n", "").strip()
    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def find_missing_superpass_inputs(workflow_text: str) -> list[str]:
    if not _normalize_optional(workflow_text):
        return []
    required = [
        "hybrid_superpass_enable",
        "hybrid_superpass_missing_mode",
        "hybrid_superpass_fail_on_failed",
    ]
    missing: list[str] = []
    for key in required:
        if key not in workflow_text:
            missing.append(key)
    return missing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch evaluation-report workflow with hybrid superpass settings and "
            "optionally watch result."
        )
    )
    parser.add_argument("--workflow", default="evaluation-report.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument(
        "--repo", default="", help="Optional owner/repo for gh commands."
    )
    parser.add_argument(
        "--hybrid-superpass-enable",
        default="true",
        help="workflow_dispatch.hybrid_superpass_enable (true/false)",
    )
    parser.add_argument(
        "--hybrid-superpass-missing-mode",
        default="fail",
        choices=("skip", "fail"),
        help="workflow_dispatch.hybrid_superpass_missing_mode",
    )
    parser.add_argument(
        "--hybrid-superpass-fail-on-failed",
        default="true",
        help="workflow_dispatch.hybrid_superpass_fail_on_failed (true/false)",
    )
    parser.add_argument("--hybrid-blind-enable", default="")
    parser.add_argument("--hybrid-blind-dxf-dir", default="")
    parser.add_argument("--hybrid-blind-fail-on-gate-failed", default="")
    parser.add_argument("--hybrid-blind-strict-require-real-data", default="")
    parser.add_argument("--hybrid-calibration-enable", default="")
    parser.add_argument("--hybrid-calibration-input-csv", default="")
    parser.add_argument(
        "--expected-conclusion",
        default="success",
        choices=("success", "failure", "cancelled"),
        help="Expected workflow run conclusion.",
    )
    parser.add_argument("--wait-timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=int, default=3)
    parser.add_argument("--list-limit", type=int, default=20)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print commands without dispatching workflows.",
    )
    parser.add_argument(
        "--skip-remote-input-check",
        action="store_true",
        help="Skip pre-dispatch check for required superpass workflow inputs on remote.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch_cmd = build_workflow_run_command(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        hybrid_superpass_enable=str(args.hybrid_superpass_enable),
        hybrid_superpass_missing_mode=str(args.hybrid_superpass_missing_mode),
        hybrid_superpass_fail_on_failed=str(args.hybrid_superpass_fail_on_failed),
        hybrid_blind_enable=str(args.hybrid_blind_enable),
        hybrid_blind_dxf_dir=str(args.hybrid_blind_dxf_dir),
        hybrid_blind_fail_on_gate_failed=str(args.hybrid_blind_fail_on_gate_failed),
        hybrid_blind_strict_require_real_data=str(
            args.hybrid_blind_strict_require_real_data
        ),
        hybrid_calibration_enable=str(args.hybrid_calibration_enable),
        hybrid_calibration_input_csv=str(args.hybrid_calibration_input_csv),
    )

    watch_hint_cmd = ["gh", "run", "watch", "<run_id>", "--exit-status"]
    view_hint_cmd = ["gh", "run", "view", "<run_id>", "--json", "conclusion,url"]
    if str(args.repo).strip():
        watch_hint_cmd.extend(["--repo", str(args.repo)])
        view_hint_cmd.extend(["--repo", str(args.repo)])
    watch_hint = shlex.join(watch_hint_cmd)
    view_hint = shlex.join(view_hint_cmd)

    print("dispatch_command=" + shlex.join(dispatch_cmd))
    print("watch_hint=" + watch_hint)
    print("view_hint=" + view_hint)

    if args.print_only:
        payload = {
            "mode": "print_only",
            "dispatch_command": dispatch_cmd,
            "watch_hint": watch_hint,
            "view_hint": view_hint,
        }
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 0

    ready, message = check_gh_ready()
    if not ready:
        print(message)
        payload = {"overall_exit_code": 1, "reason": message}
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 1

    if _normalize_optional(str(args.repo)) and not bool(args.skip_remote_input_check):
        workflow_text = fetch_remote_workflow_text(
            str(args.repo), str(args.workflow), str(args.ref)
        )
        missing_inputs = find_missing_superpass_inputs(workflow_text)
        if missing_inputs:
            print(
                "error: remote workflow is missing required superpass inputs: "
                + ", ".join(missing_inputs)
            )
            print(
                "error: sync .github/workflows/evaluation-report.yml to remote branch "
                "or re-run with --skip-remote-input-check."
            )
            payload = {
                "overall_exit_code": 1,
                "reason": "missing_remote_workflow_inputs",
                "missing_inputs": missing_inputs,
            }
            if args.output_json:
                _write_output_json(str(args.output_json), payload)
            return 1

    known_runs = list_dispatched_run_ids(
        str(args.workflow), str(args.ref), str(args.repo), limit=int(args.list_limit)
    )

    dispatch_result = subprocess.run(
        dispatch_cmd, capture_output=True, text=True, check=False
    )
    dispatch_output = (dispatch_result.stdout or dispatch_result.stderr or "").strip()
    if dispatch_result.returncode != 0:
        msg = _extract_short_error(dispatch_result, "failed to dispatch workflow")
        print(f"error: {msg}")
        if (
            "Unexpected inputs provided:" in dispatch_output
            and "hybrid_superpass_" in dispatch_output
        ):
            print(
                "error: remote workflow does not recognize hybrid superpass inputs. "
                "Sync .github/workflows/evaluation-report.yml first."
            )
        payload = {
            "overall_exit_code": 1,
            "dispatch_exit_code": int(dispatch_result.returncode),
            "dispatch_stdout": dispatch_result.stdout or "",
            "dispatch_stderr": dispatch_result.stderr or "",
            "dispatch_command": dispatch_cmd,
        }
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
    )
    if run_id is None:
        print("error: timed out waiting for dispatched workflow run id.")
        payload = {
            "overall_exit_code": 1,
            "dispatch_command": dispatch_cmd,
            "reason": "run_id_timeout",
        }
        if args.output_json:
            _write_output_json(str(args.output_json), payload)
        return 1

    watch_exit_code = watch_run(run_id, str(args.repo))
    conclusion, run_url = wait_for_run_conclusion(
        run_id=int(run_id),
        repo=str(args.repo),
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
    )
    matched = conclusion == str(args.expected_conclusion)
    overall_exit_code = 0 if matched else 1

    payload = {
        "dispatch_command": dispatch_cmd,
        "workflow": str(args.workflow),
        "ref": str(args.ref),
        "repo": str(args.repo),
        "run_id": int(run_id),
        "run_url": run_url,
        "conclusion": conclusion,
        "expected_conclusion": str(args.expected_conclusion),
        "matched_expectation": bool(matched),
        "watch_exit_code": int(watch_exit_code),
        "overall_exit_code": int(overall_exit_code),
    }
    if args.output_json:
        _write_output_json(str(args.output_json), payload)

    print(
        "result run_id={} conclusion={} expected={} matched_expectation={} watch_exit_code={}".format(
            run_id, conclusion, args.expected_conclusion, matched, watch_exit_code
        )
    )
    if run_url:
        print(f"run_url={run_url}")
    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
