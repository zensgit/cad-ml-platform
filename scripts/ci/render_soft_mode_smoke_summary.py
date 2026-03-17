#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    from scripts.ci.summary_render_utils import (
        append_markdown_section,
        boolish,
        is_zeroish,
        read_json_object,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ci.summary_render_utils import (
        append_markdown_section,
        boolish,
        is_zeroish,
        read_json_object,
    )


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
        if is_zeroish(summary.get("overall_exit_code", 1))
        and boolish(summary.get("soft_marker_ok", False))
        and boolish(summary.get("restore_ok", False))
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
        if not is_zeroish(item.get("dispatch_exit_code", 0))
        or not boolish(item.get("soft_marker_ok", False))
    ]
    last_attempt_message = "n/a"
    if attempts:
        last_attempt = attempts[-1]
        last_attempt_message = str(
            last_attempt.get("soft_marker_message", last_attempt.get("message", "n/a"))
        )
    pr_comment_error = str(pr_comment_obj.get("error") or "").strip()

    lines = ["## Evaluation Soft-Mode Smoke"]
    append_markdown_section(
        lines,
        "Smoke Verdict",
        [
            ("verdict", verdict),
            ("soft_marker_ok", summary.get("soft_marker_ok", "n/a")),
            ("restore_ok", summary.get("restore_ok", "n/a")),
            ("pr_comment_status", pr_comment_status),
            ("overall_exit_code", summary.get("overall_exit_code", "n/a")),
            ("dispatch_exit_code", summary.get("dispatch_exit_code", "n/a")),
            ("max_dispatch_attempts", summary.get("max_dispatch_attempts", "n/a")),
            ("retry_sleep_seconds", summary.get("retry_sleep_seconds", "n/a")),
            ("attempts_total", len(attempts)),
        ],
    )
    append_markdown_section(
        lines,
        "Smoke Snapshot",
        [
            ("failed_attempt_count", len(failed_attempts)),
            ("last_attempt_message", last_attempt_message),
            ("pr_comment_error", pr_comment_error or "(none)"),
        ],
    )
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
        summary = read_json_object(summary_path, "summary")
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
