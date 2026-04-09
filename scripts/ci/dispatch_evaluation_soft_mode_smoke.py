#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci import dispatch_hybrid_superpass_workflow as dispatcher
from scripts.ci import post_soft_mode_smoke_pr_comment as pr_comment


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _extract_short_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    text = (result.stderr or result.stdout or "").strip()
    if not text:
        return fallback
    return text.splitlines()[0]


def _find_variable_value(payload: Any, name: str) -> tuple[bool, str]:
    if not isinstance(payload, list):
        return (False, "")
    target = str(name or "").strip()
    if not target:
        return (False, "")
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or "").strip() != target:
            continue
        return (True, str(item.get("value") or ""))
    return (False, "")


def get_repo_variable(repo: str, name: str) -> tuple[bool, str, str]:
    result = _run(["gh", "variable", "list", "--repo", repo, "--json", "name,value"])
    if result.returncode != 0:
        return (
            False,
            "",
            f"gh variable list failed: {_extract_short_error(result, 'unknown error')}",
        )
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return (False, "", "gh variable list returned invalid JSON")
    found, value = _find_variable_value(payload, name)
    return (found, value, "")


def set_repo_variable(repo: str, name: str, value: str) -> tuple[bool, str]:
    result = _run(["gh", "variable", "set", name, "--repo", repo, "--body", value])
    if result.returncode != 0:
        return (
            False,
            f"gh variable set {name} failed: {_extract_short_error(result, 'unknown error')}",
        )
    return (True, "")


def delete_repo_variable(repo: str, name: str) -> tuple[bool, str]:
    result = _run(["gh", "variable", "delete", name, "--repo", repo])
    if result.returncode != 0:
        return (
            False,
            f"gh variable delete {name} failed: {_extract_short_error(result, 'unknown error')}",
        )
    return (True, "")


def detect_soft_mode_marker(run_id: int, repo: str) -> tuple[bool, str]:
    result = _run(["gh", "run", "view", str(run_id), "--repo", repo, "--log"])
    if result.returncode != 0:
        return (
            False,
            f"gh run view --log failed: {_extract_short_error(result, 'unknown error')}",
        )
    marker = "Resolved strict fail mode: soft"
    if marker in (result.stdout or ""):
        return (True, "")
    return (False, f"missing marker: {marker}")


def _normalize_branch_ref(ref: str) -> str:
    branch = str(ref or "").strip()
    for prefix in ("refs/heads/", "origin/"):
        if branch.startswith(prefix):
            branch = branch[len(prefix) :]
    return branch


def _resolve_current_branch() -> tuple[bool, str, str]:
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if result.returncode != 0:
        return (
            False,
            "",
            f"git rev-parse --abbrev-ref HEAD failed: {_extract_short_error(result, 'unknown error')}",
        )
    branch = str(result.stdout or "").strip()
    if not branch or branch == "HEAD":
        return (False, "", "unable to resolve current git branch")
    return (True, branch, "")


def _resolve_current_commit_sha() -> tuple[str, str]:
    result = _run(["git", "rev-parse", "HEAD"])
    if result.returncode != 0:
        return (
            "",
            f"git rev-parse HEAD failed: {_extract_short_error(result, 'unknown error')}",
        )
    sha = str(result.stdout or "").strip()
    if not sha:
        return ("", "git rev-parse HEAD returned empty sha")
    return (sha, "")


def _resolve_open_pr_number(repo: str, head_branch: str) -> tuple[int, str]:
    branch = _normalize_branch_ref(head_branch)
    if not branch:
        return (0, "head branch is empty")
    result = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            str(repo),
            "--state",
            "open",
            "--head",
            branch,
            "--json",
            "number",
            "--limit",
            "1",
        ]
    )
    if result.returncode != 0:
        return (
            0,
            f"gh pr list failed: {_extract_short_error(result, 'unknown error')}",
        )
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return (0, "gh pr list returned invalid JSON")
    if not isinstance(payload, list) or not payload:
        return (0, f"no open PR found for head branch: {branch}")
    first = payload[0]
    if not isinstance(first, dict):
        return (0, "gh pr list returned malformed item")
    try:
        return (int(first.get("number") or 0), "")
    except (TypeError, ValueError):
        return (0, "gh pr list returned non-integer PR number")


def resolve_comment_pr_number(
    *,
    repo: str,
    ref: str,
    explicit_pr_number: int,
    auto_resolve: bool,
) -> tuple[int, str, str]:
    if int(explicit_pr_number or 0) > 0:
        return (int(explicit_pr_number), "", "")
    if not bool(auto_resolve):
        return (0, "", "")
    head_branch = _normalize_branch_ref(ref)
    if head_branch in ("", "main", "master"):
        ok, current_branch, branch_err = _resolve_current_branch()
        if not ok:
            return (0, "", branch_err)
        head_branch = current_branch
    pr_number, pr_err = _resolve_open_pr_number(str(repo), head_branch)
    return (int(pr_number), str(head_branch), str(pr_err))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily set EVALUATION_STRICT_FAIL_MODE=soft, dispatch "
            "evaluation-report workflow, then restore variable."
        )
    )
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. owner/repo")
    parser.add_argument("--workflow", default="evaluation-report.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument(
        "--strict-mode-var",
        default="EVALUATION_STRICT_FAIL_MODE",
        help="GitHub repository variable controlling strict gate mode.",
    )
    parser.add_argument(
        "--soft-value",
        default="soft",
        help="Value written into strict-mode variable during smoke run.",
    )
    parser.add_argument(
        "--keep-soft",
        action="store_true",
        help="Do not restore variable after run (default restores automatically).",
    )
    parser.add_argument(
        "--skip-log-check",
        action="store_true",
        help="Skip checking run logs for strict-mode soft marker.",
    )
    parser.add_argument("--hybrid-superpass-enable", default="true")
    parser.add_argument(
        "--hybrid-superpass-missing-mode",
        default="fail",
        choices=("skip", "fail"),
    )
    parser.add_argument("--hybrid-superpass-fail-on-failed", default="true")
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
    )
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=3)
    parser.add_argument("--list-limit", type=int, default=30)
    parser.add_argument(
        "--max-dispatch-attempts",
        type=int,
        default=1,
        help="Maximum dispatch attempts before giving up.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=int,
        default=15,
        help="Sleep seconds between retries when dispatch or marker check fails.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--comment-pr-number",
        type=int,
        default=0,
        help="Optional PR number for posting soft-mode summary comment.",
    )
    parser.add_argument(
        "--comment-pr-auto",
        action="store_true",
        help=(
            "Auto resolve PR number from branch when --comment-pr-number is not provided."
        ),
    )
    parser.add_argument(
        "--comment-repo",
        default="",
        help="Optional repo for PR comment. Defaults to --repo when empty.",
    )
    parser.add_argument(
        "--comment-title",
        default="CAD ML Platform - Soft Mode Smoke",
        help="Comment title marker for create/update matching.",
    )
    parser.add_argument(
        "--comment-commit-sha",
        default="",
        help="Optional commit SHA shown in PR comment.",
    )
    parser.add_argument(
        "--comment-output-json",
        default="",
        help="Optional output json path for comment script result.",
    )
    parser.add_argument(
        "--comment-dry-run",
        action="store_true",
        help="Preview PR comment action without creating/updating.",
    )
    parser.add_argument(
        "--comment-fail-on-error",
        action="store_true",
        help="When set, non-zero PR comment result fails this script.",
    )
    parser.add_argument("--skip-remote-input-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ready, message = dispatcher.check_gh_ready()
    if not ready:
        print(message)
        return 1

    found, previous_value, get_err = get_repo_variable(args.repo, args.strict_mode_var)
    if get_err:
        print(get_err)
        return 1

    set_ok, set_err = set_repo_variable(args.repo, args.strict_mode_var, args.soft_value)
    if not set_ok:
        print(set_err)
        return 1

    payload: dict[str, Any] = {
        "repo": str(args.repo),
        "workflow": str(args.workflow),
        "ref": str(args.ref),
        "strict_mode_var": str(args.strict_mode_var),
        "soft_value": str(args.soft_value),
        "variable_found_before": bool(found),
        "variable_value_before": str(previous_value),
        "keep_soft": bool(args.keep_soft),
    }

    restore_attempted = False
    restore_ok = True
    restore_message = ""
    dispatch_exit = 1
    marker_ok = False
    marker_message = "not_checked"
    attempts_payload: list[dict[str, Any]] = []
    max_attempts = max(1, int(args.max_dispatch_attempts))
    retry_sleep_seconds = max(0, int(args.retry_sleep_seconds))
    payload["max_dispatch_attempts"] = int(max_attempts)
    payload["retry_sleep_seconds"] = int(retry_sleep_seconds)

    with tempfile.TemporaryDirectory(prefix="eval_soft_smoke_") as tmpdir:
        try:
            success_reached = False
            for attempt_index in range(1, max_attempts + 1):
                attempt_output_json = Path(tmpdir) / f"dispatch_attempt_{attempt_index}.json"
                dispatch_args = [
                    "--workflow",
                    str(args.workflow),
                    "--ref",
                    str(args.ref),
                    "--repo",
                    str(args.repo),
                    "--hybrid-superpass-enable",
                    str(args.hybrid_superpass_enable),
                    "--hybrid-superpass-missing-mode",
                    str(args.hybrid_superpass_missing_mode),
                    "--hybrid-superpass-fail-on-failed",
                    str(args.hybrid_superpass_fail_on_failed),
                    "--hybrid-blind-enable",
                    str(args.hybrid_blind_enable),
                    "--hybrid-blind-dxf-dir",
                    str(args.hybrid_blind_dxf_dir),
                    "--hybrid-blind-fail-on-gate-failed",
                    str(args.hybrid_blind_fail_on_gate_failed),
                    "--hybrid-blind-strict-require-real-data",
                    str(args.hybrid_blind_strict_require_real_data),
                    "--hybrid-calibration-enable",
                    str(args.hybrid_calibration_enable),
                    "--hybrid-calibration-input-csv",
                    str(args.hybrid_calibration_input_csv),
                    "--expected-conclusion",
                    str(args.expected_conclusion),
                    "--wait-timeout-seconds",
                    str(int(args.wait_timeout_seconds)),
                    "--poll-interval-seconds",
                    str(int(args.poll_interval_seconds)),
                    "--list-limit",
                    str(int(args.list_limit)),
                    "--output-json",
                    str(attempt_output_json),
                ]
                if bool(args.skip_remote_input_check):
                    dispatch_args.append("--skip-remote-input-check")

                attempt_entry: dict[str, Any] = {
                    "attempt": int(attempt_index),
                }
                dispatch_exit = dispatcher.main(dispatch_args)
                attempt_entry["dispatch_exit_code"] = int(dispatch_exit)
                dispatch_payload: dict[str, Any] = {}
                if attempt_output_json.exists():
                    dispatch_payload = json.loads(
                        attempt_output_json.read_text(encoding="utf-8")
                    )
                    attempt_entry["dispatch"] = dispatch_payload

                run_id_value = (
                    dispatch_payload.get("run_id")
                    if isinstance(dispatch_payload, dict)
                    else None
                )
                if run_id_value and not bool(args.skip_log_check):
                    marker_ok, marker_message = detect_soft_mode_marker(
                        int(run_id_value), str(args.repo)
                    )
                elif bool(args.skip_log_check):
                    marker_ok = True
                    marker_message = "skipped"
                else:
                    marker_ok = False
                    marker_message = "run_id_missing"

                attempt_entry["soft_marker_ok"] = bool(marker_ok)
                attempt_entry["soft_marker_message"] = str(marker_message)
                attempts_payload.append(attempt_entry)

                if int(dispatch_exit) == 0 and bool(marker_ok):
                    payload["dispatch_exit_code"] = int(dispatch_exit)
                    payload["dispatch"] = dispatch_payload
                    success_reached = True
                    break

                if attempt_index < max_attempts and retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds)

            if not success_reached:
                last_entry = attempts_payload[-1] if attempts_payload else {}
                payload["dispatch_exit_code"] = int(last_entry.get("dispatch_exit_code", 1))
                last_dispatch = last_entry.get("dispatch")
                if isinstance(last_dispatch, dict):
                    payload["dispatch"] = last_dispatch
                marker_ok = bool(last_entry.get("soft_marker_ok", False))
                marker_message = str(last_entry.get("soft_marker_message", "unknown"))
        finally:
            if not bool(args.keep_soft):
                restore_attempted = True
                if found:
                    restore_ok, restore_message = set_repo_variable(
                        args.repo, args.strict_mode_var, previous_value
                    )
                else:
                    restore_ok, restore_message = delete_repo_variable(
                        args.repo, args.strict_mode_var
                    )

    payload["soft_marker_ok"] = bool(marker_ok)
    payload["soft_marker_message"] = str(marker_message)
    payload["attempts"] = attempts_payload
    payload["restore_attempted"] = bool(restore_attempted)
    payload["restore_ok"] = bool(restore_ok)
    payload["restore_message"] = str(restore_message)

    comment_pr_number, comment_pr_branch, comment_pr_resolve_error = resolve_comment_pr_number(
        repo=str(args.repo),
        ref=str(args.ref),
        explicit_pr_number=int(args.comment_pr_number or 0),
        auto_resolve=bool(args.comment_pr_auto),
    )
    comment_requested = int(args.comment_pr_number or 0) > 0 or bool(args.comment_pr_auto)
    comment_enabled = comment_pr_number > 0
    comment_repo = str(args.comment_repo or args.repo)
    comment_payload: dict[str, Any] = {
        "requested": bool(comment_requested),
        "enabled": bool(comment_enabled),
        "repo": str(comment_repo),
        "pr_number": int(comment_pr_number),
        "auto_resolve": bool(args.comment_pr_auto),
        "head_branch": str(comment_pr_branch),
        "dry_run": bool(args.comment_dry_run),
        "fail_on_error": bool(args.comment_fail_on_error),
        "exit_code": 0,
        "error": str(comment_pr_resolve_error),
    }
    comment_commit_sha = str(args.comment_commit_sha).strip()
    if comment_enabled and not comment_commit_sha:
        resolved_sha, resolved_sha_error = _resolve_current_commit_sha()
        if resolved_sha:
            comment_commit_sha = str(resolved_sha)
        elif not comment_payload["error"]:
            comment_payload["error"] = str(resolved_sha_error)
    if comment_enabled:
        with tempfile.TemporaryDirectory(prefix="eval_soft_comment_") as comment_tmpdir:
            comment_summary_json = (
                Path(comment_tmpdir) / "soft_mode_smoke_summary_for_comment.json"
            )
            comment_summary_json.write_text(
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
                encoding="utf-8",
            )
            comment_args = [
                "--repo",
                str(comment_repo),
                "--pr-number",
                str(comment_pr_number),
                "--summary-json",
                str(comment_summary_json),
                "--title",
                str(args.comment_title),
                "--commit-sha",
                str(comment_commit_sha),
            ]
            comment_output_json = str(args.comment_output_json or "").strip()
            if comment_output_json:
                comment_args.extend(["--output-json", comment_output_json])
            if bool(args.comment_dry_run):
                comment_args.append("--dry-run")
            try:
                comment_rc = int(pr_comment.main(comment_args))
                comment_payload["exit_code"] = int(comment_rc)
            except Exception as exc:  # pragma: no cover - defensive guard
                comment_payload["exit_code"] = 1
                comment_payload["error"] = str(exc)
    payload["pr_comment"] = comment_payload

    overall_exit = 0 if int(dispatch_exit) == 0 else 1
    if not marker_ok:
        overall_exit = 1
    if restore_attempted and not restore_ok:
        overall_exit = 1
    if (
        comment_requested
        and bool(args.comment_fail_on_error)
        and (
            int(comment_payload.get("exit_code", 0)) != 0
            or (
                not comment_enabled
                and bool(str(comment_payload.get("error") or "").strip())
            )
        )
    ):
        overall_exit = 1
    payload["overall_exit_code"] = int(overall_exit)

    if args.output_json:
        out_path = Path(args.output_json).expanduser()
        if out_path.parent != Path("."):
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
        )

    print(
        "result overall_exit_code={} dispatch_exit_code={} soft_marker_ok={} restore_ok={}".format(
            payload["overall_exit_code"],
            payload.get("dispatch_exit_code", 1),
            payload["soft_marker_ok"],
            payload["restore_ok"],
        )
    )
    dispatch_payload = payload.get("dispatch") if isinstance(payload.get("dispatch"), dict) else {}
    run_id = dispatch_payload.get("run_id", "")
    run_url = dispatch_payload.get("run_url", "")
    if run_id:
        print(f"run_id={run_id}")
    if run_url:
        print(f"run_url={run_url}")
    if payload["soft_marker_message"]:
        print(f"soft_marker_message={payload['soft_marker_message']}")
    if restore_attempted:
        print(f"restore_message={restore_message or 'ok'}")
    if comment_enabled:
        print(
            "pr_comment exit_code={} repo={} pr_number={} dry_run={}".format(
                comment_payload.get("exit_code", 0),
                comment_payload.get("repo", ""),
                comment_payload.get("pr_number", 0),
                comment_payload.get("dry_run", False),
            )
        )
    if comment_payload.get("error"):
        print(f"pr_comment_error={comment_payload['error']}")

    return int(overall_exit)


if __name__ == "__main__":
    raise SystemExit(main())
