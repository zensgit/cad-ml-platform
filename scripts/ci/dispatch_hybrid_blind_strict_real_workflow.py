#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Sequence

_NON_FAILING_CONCLUSIONS = {"success", "skipped", "neutral"}


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
    repo: str,
    hybrid_blind_dxf_dir: str,
    hybrid_blind_manifest_csv: str,
    hybrid_blind_synth_manifest: str,
    strict_fail_on_gate_failed: str,
    strict_require_real_data: str,
) -> list[str]:
    command = [
        "gh",
        "workflow",
        "run",
        str(workflow),
        "--ref",
        str(ref),
        "-f",
        "hybrid_blind_enable=true",
        "-f",
        f"hybrid_blind_dxf_dir={hybrid_blind_dxf_dir}",
        "-f",
        f"hybrid_blind_fail_on_gate_failed={strict_fail_on_gate_failed}",
        "-f",
        f"hybrid_blind_strict_require_real_data={strict_require_real_data}",
    ]
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
    if hybrid_blind_manifest_csv:
        command.extend(["-f", f"hybrid_blind_manifest_csv={hybrid_blind_manifest_csv}"])
    if hybrid_blind_synth_manifest:
        command.extend(["-f", f"hybrid_blind_synth_manifest={hybrid_blind_synth_manifest}"])
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
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
    return command


def list_dispatched_run_ids(workflow: str, ref: str, repo: str, *, limit: int) -> list[int]:
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
) -> int | None:
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
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
    result = subprocess.run(command, check=False)
    return int(result.returncode)


def get_run_conclusion_and_url(run_id: int, repo: str) -> tuple[str, str]:
    command = ["gh", "run", "view", str(run_id), "--json", "conclusion,url"]
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
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


def _is_failed_conclusion(raw_value: Any) -> bool:
    conclusion = str(raw_value or "").strip().lower()
    if not conclusion:
        return False
    return conclusion not in _NON_FAILING_CONCLUSIONS


def summarize_failed_jobs(
    jobs_payload: Any,
    *,
    max_jobs: int = 5,
) -> dict[str, Any]:
    if not isinstance(jobs_payload, list):
        return {
            "total_jobs": 0,
            "failed_job_count": 0,
            "failed_jobs": [],
            "failed_jobs_truncated": False,
        }

    failed_jobs: list[dict[str, str]] = []
    failed_job_count = 0
    total_jobs = 0
    max_items = max(1, int(max_jobs))
    for item in jobs_payload:
        if not isinstance(item, dict):
            continue
        total_jobs += 1
        job_conclusion = str(item.get("conclusion") or "")
        if not _is_failed_conclusion(job_conclusion):
            continue
        failed_job_count += 1

        first_failed_step_name = ""
        first_failed_step_conclusion = ""
        steps_payload = item.get("steps")
        if isinstance(steps_payload, list):
            for step in steps_payload:
                if not isinstance(step, dict):
                    continue
                step_conclusion = str(step.get("conclusion") or "")
                if _is_failed_conclusion(step_conclusion):
                    first_failed_step_name = str(step.get("name") or "")
                    first_failed_step_conclusion = step_conclusion
                    break

        if len(failed_jobs) < max_items:
            failed_jobs.append(
                {
                    "job_name": str(item.get("name") or ""),
                    "job_conclusion": job_conclusion,
                    "job_url": str(item.get("url") or ""),
                    "failed_step_name": first_failed_step_name,
                    "failed_step_conclusion": first_failed_step_conclusion,
                }
            )

    return {
        "total_jobs": total_jobs,
        "failed_job_count": failed_job_count,
        "failed_jobs": failed_jobs,
        "failed_jobs_truncated": len(failed_jobs) < failed_job_count,
    }


def fetch_run_failure_diagnostics(run_id: int, repo: str) -> dict[str, Any]:
    command = ["gh", "run", "view", str(run_id), "--json", "jobs"]
    if str(repo).strip():
        command.extend(["--repo", str(repo).strip()])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "reason": "gh_run_view_failed",
            "stderr": (result.stderr or "").strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "available": False,
            "reason": "gh_run_view_json_decode_failed",
        }
    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": "gh_run_view_invalid_payload",
        }
    summary = summarize_failed_jobs(payload.get("jobs"))
    return {
        "available": True,
        "reason": "",
        **summary,
    }


def fetch_remote_workflow_text(repo: str, workflow: str, ref: str) -> str:
    repo_text = str(repo or "").strip()
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


def find_missing_strict_real_inputs(workflow_text: str) -> list[str]:
    if not str(workflow_text or "").strip():
        return []
    required = [
        "hybrid_blind_enable",
        "hybrid_blind_dxf_dir",
        "hybrid_blind_fail_on_gate_failed",
        "hybrid_blind_strict_require_real_data",
    ]
    missing: list[str] = []
    for key in required:
        if key not in workflow_text:
            missing.append(key)
    return missing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch evaluation-report workflow with strict real-data hybrid blind settings "
            "and optionally watch result."
        )
    )
    parser.add_argument("--workflow", default="evaluation-report.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--repo", default="", help="Optional owner/repo for gh commands.")
    parser.add_argument(
        "--hybrid-blind-dxf-dir",
        required=True,
        help="workflow_dispatch.hybrid_blind_dxf_dir value (path visible in CI runner)",
    )
    parser.add_argument(
        "--hybrid-blind-manifest-csv",
        default="",
        help="Optional workflow_dispatch.hybrid_blind_manifest_csv value.",
    )
    parser.add_argument(
        "--hybrid-blind-synth-manifest",
        default="",
        help="Optional workflow_dispatch.hybrid_blind_synth_manifest value.",
    )
    parser.add_argument(
        "--strict-fail-on-gate-failed",
        default="true",
        help="workflow_dispatch.hybrid_blind_fail_on_gate_failed (true/false)",
    )
    parser.add_argument(
        "--strict-require-real-data",
        default="true",
        help="workflow_dispatch.hybrid_blind_strict_require_real_data (true/false)",
    )
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
        help="Skip pre-dispatch check for required strict-real workflow inputs on remote.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dispatch_cmd = build_workflow_run_command(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        hybrid_blind_dxf_dir=str(args.hybrid_blind_dxf_dir),
        hybrid_blind_manifest_csv=str(args.hybrid_blind_manifest_csv),
        hybrid_blind_synth_manifest=str(args.hybrid_blind_synth_manifest),
        strict_fail_on_gate_failed=str(args.strict_fail_on_gate_failed),
        strict_require_real_data=str(args.strict_require_real_data),
    )
    list_cmd = _build_list_dispatched_runs_command(
        str(args.workflow), str(args.ref), str(args.repo), limit=int(args.list_limit)
    )

    if args.print_only:
        print("# dispatch command")
        print(shlex.join(dispatch_cmd))
        print("# list dispatched runs command")
        print(shlex.join(list_cmd))
        print("# watch command")
        print("gh run watch <run_id> --exit-status")
        print("# view command")
        print("gh run view <run_id> --json conclusion,url")
        return 0

    ready, reason = check_gh_ready()
    if not ready:
        print(f"error: {reason}")
        return 1

    if str(args.repo or "").strip() and not bool(args.skip_remote_input_check):
        workflow_text = fetch_remote_workflow_text(
            str(args.repo), str(args.workflow), str(args.ref)
        )
        missing_inputs = find_missing_strict_real_inputs(workflow_text)
        if missing_inputs:
            joined = ", ".join(missing_inputs)
            print(
                "error: remote workflow is missing required strict-real inputs: "
                f"{joined}. Merge/push workflow changes first, then retry dispatch."
            )
            return 1

    known_run_ids = list_dispatched_run_ids(
        str(args.workflow), str(args.ref), str(args.repo), limit=int(args.list_limit)
    )
    dispatch_result = subprocess.run(dispatch_cmd, capture_output=True, text=True, check=False)
    dispatch_exit_code = int(dispatch_result.returncode)
    if dispatch_exit_code != 0:
        dispatch_error = _extract_short_error(dispatch_result, "gh workflow run failed")
        if "Unexpected inputs provided" in (dispatch_result.stderr or dispatch_result.stdout or ""):
            print(
                "error: remote workflow does not recognize hybrid blind strict-real inputs. "
                "Merge/push workflow changes first, then retry dispatch."
            )
            print(f"details: {dispatch_error}")
            return 1
        print(
            "error: failed to dispatch workflow: "
            f"{dispatch_error}"
        )
        return 1

    run_id = wait_for_new_dispatched_run_id(
        workflow=str(args.workflow),
        ref=str(args.ref),
        repo=str(args.repo),
        known_run_ids=known_run_ids,
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
        list_limit=int(args.list_limit),
    )
    if run_id is None:
        print("error: timed out waiting for a new workflow_dispatch run id.")
        return 1

    watch_exit_code = watch_run(int(run_id), str(args.repo))
    conclusion, run_url = wait_for_run_conclusion(
        run_id=int(run_id),
        repo=str(args.repo),
        timeout_seconds=int(args.wait_timeout_seconds),
        poll_interval_seconds=int(args.poll_interval_seconds),
    )
    matched_expectation = bool(conclusion and conclusion == str(args.expected_conclusion))
    overall_exit_code = 0 if matched_expectation else 1

    payload = {
        "workflow": str(args.workflow),
        "ref": str(args.ref),
        "repo": str(args.repo),
        "hybrid_blind_dxf_dir": str(args.hybrid_blind_dxf_dir),
        "hybrid_blind_manifest_csv": str(args.hybrid_blind_manifest_csv),
        "hybrid_blind_synth_manifest": str(args.hybrid_blind_synth_manifest),
        "strict_fail_on_gate_failed": str(args.strict_fail_on_gate_failed),
        "strict_require_real_data": str(args.strict_require_real_data),
        "expected_conclusion": str(args.expected_conclusion),
        "run_id": int(run_id),
        "run_url": run_url,
        "conclusion": conclusion,
        "dispatch_exit_code": dispatch_exit_code,
        "watch_exit_code": int(watch_exit_code),
        "matched_expectation": matched_expectation,
        "overall_exit_code": overall_exit_code,
    }
    if overall_exit_code != 0:
        payload["failure_diagnostics"] = fetch_run_failure_diagnostics(
            int(run_id), str(args.repo)
        )
        diagnostics = payload["failure_diagnostics"]
        has_failed_jobs = bool(diagnostics.get("available")) and int(
            diagnostics.get("failed_job_count") or 0
        ) > 0
        if has_failed_jobs:
            first_failed = (diagnostics.get("failed_jobs") or [{}])[0]
            print(
                (
                    "diagnostic first_failed_job={} job_conclusion={} failed_step={} "
                    "step_conclusion={}"
                ).format(
                    first_failed.get("job_name") or "",
                    first_failed.get("job_conclusion") or "",
                    first_failed.get("failed_step_name") or "",
                    first_failed.get("failed_step_conclusion") or "",
                )
            )
    if args.output_json:
        _write_output_json(str(args.output_json), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(overall_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
