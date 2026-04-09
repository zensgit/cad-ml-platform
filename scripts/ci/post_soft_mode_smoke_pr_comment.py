#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.ci.comment_markdown_utils import (
        markdown_footer,
        markdown_section,
        markdown_table,
    )
    from scripts.ci.summary_render_utils import read_json_object
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ci.comment_markdown_utils import (
        markdown_footer,
        markdown_section,
        markdown_table,
    )
    from scripts.ci.summary_render_utils import read_json_object


def _run(command: list[str], input_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def _extract_short_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    text = (result.stderr or result.stdout or "").strip()
    if not text:
        return fallback
    return text.splitlines()[0]


def _read_summary(path: Path) -> dict[str, Any]:
    return read_json_object(path, "summary")


def _normalize_attempt_rows(summary: dict[str, Any]) -> list[str]:
    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return ["- attempts: none"]
    rows: list[str] = []
    for index, item in enumerate(attempts, start=1):
        attempt = item if isinstance(item, dict) else {}
        attempt_no = _normalize_value(attempt.get("attempt", index), index)
        dispatch_exit = _normalize_value(attempt.get("dispatch_exit_code", "n/a"), "n/a")
        marker_ok = _normalize_value(attempt.get("soft_marker_ok", "n/a"), "n/a")
        message = _normalize_value(
            attempt.get("soft_marker_message", attempt.get("message", "n/a")),
            "n/a",
        )
        rows.append(
            f"- attempt {attempt_no}: dispatch_exit_code={dispatch_exit}, "
            f"soft_marker_ok={marker_ok}, message={message}"
        )
    return rows


def _normalize_value(value: Any, fallback: Any) -> str:
    if value is None or value == "":
        return str(fallback)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_comment_body(
    *,
    summary: dict[str, Any],
    title: str,
    commit_sha: str,
    updated_at: str = "",
) -> str:
    dispatch = summary.get("dispatch")
    dispatch_obj = dispatch if isinstance(dispatch, dict) else {}
    run_id = _normalize_value(dispatch_obj.get("run_id", "n/a"), "n/a")
    run_url = _normalize_value(dispatch_obj.get("run_url", "n/a"), "n/a")
    attempts_total = 0
    attempts = summary.get("attempts")
    if isinstance(attempts, list):
        attempts_total = len(attempts)
    short_sha = _normalize_value(str(commit_sha or "").strip()[:7], "n/a")
    attempt_rows = "\n".join(_normalize_attempt_rows(summary))
    return "\n".join(
        [
            f"## {title}",
            "",
            markdown_table(
                ["Field", "Value"],
                [
                    ["overall_exit_code", _normalize_value(summary.get("overall_exit_code", "n/a"), "n/a")],
                    ["dispatch_exit_code", _normalize_value(summary.get("dispatch_exit_code", "n/a"), "n/a")],
                    ["soft_marker_ok", _normalize_value(summary.get("soft_marker_ok", "n/a"), "n/a")],
                    ["restore_ok", _normalize_value(summary.get("restore_ok", "n/a"), "n/a")],
                    ["run_id", run_id],
                    ["run_url", run_url],
                    ["attempts_total", attempts_total],
                ],
            ),
            "",
            markdown_section("Attempts", attempt_rows),
            "",
            markdown_footer(updated_at=updated_at, commit_sha=short_sha),
        ]
    )


def _list_pr_comments(repo: str, pr_number: int) -> list[dict[str, Any]]:
    result = _run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{int(pr_number)}/comments",
            "--paginate",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            "gh api list comments failed: "
            + _extract_short_error(result, "unknown error")
        )
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh api list comments returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise RuntimeError("gh api list comments payload is not a list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _find_existing_comment_id(comments: list[dict[str, Any]], title: str) -> int:
    token = str(title or "").strip()
    if not token:
        return 0
    for item in comments:
        user = item.get("user")
        user_obj = user if isinstance(user, dict) else {}
        user_type = str(user_obj.get("type") or "")
        body = str(item.get("body") or "")
        comment_id = item.get("id")
        if user_type == "Bot" and token in body:
            try:
                return int(comment_id)
            except (TypeError, ValueError):
                continue
    return 0


def _create_comment(repo: str, pr_number: int, body: str) -> int:
    result = _run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{int(pr_number)}/comments",
            "--method",
            "POST",
            "--input",
            "-",
        ],
        input_text=json.dumps({"body": body}, ensure_ascii=False),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "gh api create comment failed: "
            + _extract_short_error(result, "unknown error")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh api create comment returned invalid JSON") from exc
    if not isinstance(payload, dict):
        return 0
    return int(payload.get("id") or 0)


def _update_comment(repo: str, comment_id: int, body: str) -> int:
    result = _run(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/comments/{int(comment_id)}",
            "--method",
            "PATCH",
            "--input",
            "-",
        ],
        input_text=json.dumps({"body": body}, ensure_ascii=False),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "gh api update comment failed: "
            + _extract_short_error(result, "unknown error")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh api update comment returned invalid JSON") from exc
    if not isinstance(payload, dict):
        return 0
    return int(payload.get("id") or 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update a soft-mode smoke summary comment on a PR via gh api."
        )
    )
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. owner/repo")
    parser.add_argument("--pr-number", required=True, type=int, help="Pull request number")
    parser.add_argument("--summary-json", required=True, help="Soft-mode smoke summary json path")
    parser.add_argument(
        "--title",
        default="CAD ML Platform - Soft Mode Smoke",
        help="Comment title marker used for create/update matching.",
    )
    parser.add_argument(
        "--commit-sha",
        default="",
        help="Optional commit SHA shown in comment footer.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print intended action only")
    parser.add_argument("--output-json", default="", help="Optional output result json path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    summary_path = Path(args.summary_json).expanduser()
    if not summary_path.exists():
        print(f"summary json does not exist: {summary_path}")
        return 1

    try:
        summary = _read_summary(summary_path)
        body = build_comment_body(
            summary=summary,
            title=str(args.title),
            commit_sha=str(args.commit_sha),
            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        comments = _list_pr_comments(str(args.repo), int(args.pr_number))
        existing_comment_id = _find_existing_comment_id(comments, str(args.title))
    except RuntimeError as exc:
        print(str(exc))
        return 1

    result_payload: dict[str, Any] = {
        "repo": str(args.repo),
        "pr_number": int(args.pr_number),
        "summary_json": summary_path.as_posix(),
        "dry_run": bool(args.dry_run),
        "title": str(args.title),
        "existing_comment_id": int(existing_comment_id),
    }

    if bool(args.dry_run):
        result_payload["action"] = (
            "dry_run_update_comment"
            if existing_comment_id
            else "dry_run_create_comment"
        )
        result_payload["body"] = body
    else:
        try:
            if existing_comment_id:
                updated_id = _update_comment(str(args.repo), existing_comment_id, body)
                result_payload["action"] = "update_comment"
                result_payload["updated_comment_id"] = int(updated_id or existing_comment_id)
            else:
                created_id = _create_comment(str(args.repo), int(args.pr_number), body)
                result_payload["action"] = "create_comment"
                result_payload["created_comment_id"] = int(created_id)
        except RuntimeError as exc:
            result_payload["action"] = "failed"
            result_payload["error"] = str(exc)
            if args.output_json:
                output_path = Path(str(args.output_json)).expanduser()
                if output_path.parent != Path("."):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    f"{json.dumps(result_payload, ensure_ascii=False, indent=2)}\n",
                    encoding="utf-8",
                )
            print(str(exc))
            return 1

    if args.output_json:
        output_path = Path(str(args.output_json)).expanduser()
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"{json.dumps(result_payload, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )

    print(
        "result action={} existing_comment_id={} dry_run={}".format(
            result_payload.get("action", "unknown"),
            result_payload.get("existing_comment_id", 0),
            bool(args.dry_run),
        )
    )
    if result_payload.get("action") == "create_comment":
        print(f"created_comment_id={result_payload.get('created_comment_id', 0)}")
    if result_payload.get("action") == "update_comment":
        print(f"updated_comment_id={result_payload.get('updated_comment_id', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
