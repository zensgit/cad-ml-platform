#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.ci.summary_render_utils import boolish, read_json_object, top_nonempty
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ci.summary_render_utils import boolish, read_json_object, top_nonempty


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

    lines.extend(
        [
            "",
            "## Dispatch Verdict",
            "",
            f"- verdict: {verdict}",
            f"- conclusion_pair: expected={payload.get('expected_conclusion', 'n/a')} actual={payload.get('conclusion', 'n/a')}",
            f"- top_failed_jobs: {', '.join(top_failed_jobs) if top_failed_jobs else '(none)'}",
            f"- top_failed_steps: {', '.join(top_failed_steps) if top_failed_steps else '(none)'}",
        ]
    )
    diagnostics_reason = str(diagnostics_obj.get("reason") or "").strip()
    if diagnostics_reason:
        lines.append(f"- diagnostics_reason: {diagnostics_reason}")
    elif reason:
        lines.append(f"- diagnostics_reason: {reason}")

    lines.extend(
        [
            "",
            "## Dispatch Snapshot",
            "",
            f"- failed_job_count: {len(failed_jobs_list)}",
            f"- top_failed_jobs: {', '.join(top_failed_jobs) if top_failed_jobs else '(none)'}",
            f"- top_failed_steps: {', '.join(top_failed_steps) if top_failed_steps else '(none)'}",
            f"- failure_reason: {diagnostics_reason or reason or 'n/a'}",
        ]
    )

    if diagnostics_obj:
        lines.extend(
            [
                "",
                "### Failure Diagnostics",
                f"- available: {diagnostics_obj.get('available', 'n/a')}",
                f"- failed_job_count: {diagnostics_obj.get('failed_job_count', 'n/a')}",
            ]
        )
        detail_reason = str(diagnostics_obj.get("reason") or "").strip()
        if detail_reason:
            lines.append(f"- reason: {detail_reason}")
        for item in failed_jobs_list:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- failed_job: {} job_conclusion={} failed_step={} step_conclusion={}".format(
                    item.get("job_name", ""),
                    item.get("job_conclusion", ""),
                    item.get("failed_step_name", ""),
                    item.get("failed_step_conclusion", ""),
                )
            )

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
