#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.ci.summary_render_utils import (
        append_failure_diagnostics_section,
        append_dispatch_verdict_and_snapshot_sections,
        boolish,
        read_json_object,
        render_inline_items,
        top_nonempty,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ci.summary_render_utils import (
        append_failure_diagnostics_section,
        append_dispatch_verdict_and_snapshot_sections,
        boolish,
        read_json_object,
        render_inline_items,
        top_nonempty,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("failure_diagnostics")
    diagnostics_obj = diagnostics if isinstance(diagnostics, dict) else {}
    failed_jobs = diagnostics_obj.get("failed_jobs")
    failed_jobs_list = failed_jobs if isinstance(failed_jobs, list) else []

    dispatch_command = payload.get("dispatch_command")
    dispatch_text = ""
    if isinstance(dispatch_command, list):
        dispatch_text = " ".join(str(item) for item in dispatch_command)

    verdict = (
        "matched_expectation"
        if boolish(payload.get("matched_expectation", False))
        else "expectation_mismatch"
    )
    top_failed_jobs = top_nonempty(
        item.get("job_name") if isinstance(item, dict) else ""
        for item in failed_jobs_list
    )
    top_failed_steps = top_nonempty(
        item.get("failed_step_name") if isinstance(item, dict) else ""
        for item in failed_jobs_list
    )

    lines = [
        "## Hybrid Superpass Dispatch",
        "",
        f"- workflow: {payload.get('workflow', 'n/a')}",
        f"- ref: {payload.get('ref', 'n/a')}",
        f"- repo: {payload.get('repo', 'n/a')}",
        f"- expected_conclusion: {payload.get('expected_conclusion', 'n/a')}",
        f"- conclusion: {payload.get('conclusion', 'n/a')}",
        f"- matched_expectation: {payload.get('matched_expectation', 'n/a')}",
        f"- overall_exit_code: {payload.get('overall_exit_code', 'n/a')}",
        f"- watch_exit_code: {payload.get('watch_exit_code', 'n/a')}",
        f"- run_id: {payload.get('run_id', 'n/a')}",
        f"- run_url: {payload.get('run_url', 'n/a')}",
    ]
    if dispatch_text:
        lines.append(f"- dispatch_command: {dispatch_text}")
    reason = str(payload.get("reason") or "").strip()
    if reason:
        lines.append(f"- reason: {reason}")

    top_failed_jobs_text = render_inline_items(top_failed_jobs)
    top_failed_steps_text = render_inline_items(top_failed_steps)
    append_dispatch_verdict_and_snapshot_sections(
        lines,
        verdict=verdict,
        expected_conclusion=payload.get("expected_conclusion", "n/a"),
        conclusion=payload.get("conclusion", "n/a"),
        top_failed_jobs=top_failed_jobs_text,
        top_failed_steps=top_failed_steps_text,
        failed_job_count=len(failed_jobs_list),
        diagnostics_reason=str(diagnostics_obj.get("reason") or "").strip(),
        fallback_reason=reason,
    )

    append_failure_diagnostics_section(lines, diagnostics_obj, failed_jobs_list)

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render markdown summary from hybrid superpass dispatch json."
    )
    parser.add_argument("--dispatch-json", required=True, help="Dispatch json path")
    parser.add_argument("--output-md", default="", help="Optional markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch_path = Path(str(args.dispatch_json)).expanduser()
    if not dispatch_path.exists():
        print(f"dispatch json does not exist: {dispatch_path}")
        return 1

    try:
        payload = read_json_object(dispatch_path, "dispatch")
        markdown = render_markdown(payload)
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
