#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read summary json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse summary json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("summary json must be an object")
    return payload


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _is_zeroish(value: Any) -> bool:
    return str(value if value is not None else "").strip() == "0"


def _normalize_attempts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = summary.get("attempts")
    rows: list[dict[str, Any]] = []
    if not isinstance(attempts, list):
        return rows
    for item in attempts:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    attempts = _normalize_attempts(summary)
    dispatch = summary.get("dispatch")
    dispatch_obj = dispatch if isinstance(dispatch, dict) else {}
    pr_comment = summary.get("pr_comment")
    pr_comment_obj = pr_comment if isinstance(pr_comment, dict) else {}
    verdict = (
        "ok"
        if _is_zeroish(summary.get("overall_exit_code", 1))
        and _boolish(summary.get("soft_marker_ok", False))
        and _boolish(summary.get("restore_ok", False))
        else "attention_required"
    )
    pr_comment_status = "n/a"
    if pr_comment_obj:
        pr_comment_status = (
            f"requested={pr_comment_obj.get('requested', 'n/a')}, "
            f"enabled={pr_comment_obj.get('enabled', 'n/a')}, "
            f"exit_code={pr_comment_obj.get('exit_code', 'n/a')}"
        )
    failed_attempts = [
        item
        for item in attempts
        if not _is_zeroish(item.get("dispatch_exit_code", 0))
        or not _boolish(item.get("soft_marker_ok", False))
    ]
    last_attempt_message = "n/a"
    if attempts:
        last_attempt = attempts[-1]
        last_attempt_message = str(
            last_attempt.get("soft_marker_message", last_attempt.get("message", "n/a"))
        )
    pr_comment_error = str(pr_comment_obj.get("error") or "").strip()

    lines = [
        "## Evaluation Soft-Mode Smoke",
        "",
        "## Smoke Verdict",
        "",
        f"- verdict: {verdict}",
        f"- soft_marker_ok: {summary.get('soft_marker_ok', 'n/a')}",
        f"- restore_ok: {summary.get('restore_ok', 'n/a')}",
        f"- pr_comment_status: {pr_comment_status}",
        "",
        f"- overall_exit_code: {summary.get('overall_exit_code', 'n/a')}",
        f"- dispatch_exit_code: {summary.get('dispatch_exit_code', 'n/a')}",
        f"- max_dispatch_attempts: {summary.get('max_dispatch_attempts', 'n/a')}",
        f"- retry_sleep_seconds: {summary.get('retry_sleep_seconds', 'n/a')}",
        f"- attempts_total: {len(attempts)}",
        "",
        "## Smoke Snapshot",
        "",
        f"- failed_attempt_count: {len(failed_attempts)}",
        f"- last_attempt_message: {last_attempt_message}",
        f"- pr_comment_error: {pr_comment_error or '(none)'}",
    ]
    for index, item in enumerate(attempts, start=1):
        attempt_no = item.get("attempt", index)
        dispatch_exit_code = item.get("dispatch_exit_code", "n/a")
        soft_marker_ok = item.get("soft_marker_ok", "n/a")
        message = item.get("soft_marker_message", item.get("message", "n/a"))
        lines.append(
            f"- attempt #{attempt_no}: dispatch_exit_code={dispatch_exit_code}, "
            f"soft_marker_ok={soft_marker_ok}, message={message}"
        )

    run_id = dispatch_obj.get("run_id")
    run_url = dispatch_obj.get("run_url")
    if run_id:
        lines.append(f"- run_id: {run_id}")
    if run_url:
        lines.append(f"- run_url: {run_url}")

    if pr_comment_obj:
        lines.extend(
            [
                f"- pr_comment_requested: {pr_comment_obj.get('requested', 'n/a')}",
                f"- pr_comment_enabled: {pr_comment_obj.get('enabled', 'n/a')}",
                f"- pr_comment_pr_number: {pr_comment_obj.get('pr_number', 'n/a')}",
                f"- pr_comment_auto_resolve: {pr_comment_obj.get('auto_resolve', 'n/a')}",
                f"- pr_comment_exit_code: {pr_comment_obj.get('exit_code', 'n/a')}",
            ]
        )
        if pr_comment_error:
            lines.append(f"- pr_comment_error: {pr_comment_error}")

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render markdown summary from evaluation soft-mode smoke json."
    )
    parser.add_argument("--summary-json", required=True, help="Soft-mode smoke summary json path")
    parser.add_argument("--output-md", default="", help="Optional markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary_path = Path(str(args.summary_json)).expanduser()
    if not summary_path.exists():
        print(f"summary json does not exist: {summary_path}")
        return 1

    try:
        summary = _read_summary(summary_path)
        markdown = render_markdown(summary)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    output_md = str(args.output_md or "").strip()
    if output_md:
        output_path = Path(output_md).expanduser()
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
