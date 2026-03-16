#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read dispatch json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to parse dispatch json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("dispatch json must be an object")
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("failure_diagnostics")
    diagnostics_obj = diagnostics if isinstance(diagnostics, dict) else {}
    failed_jobs = diagnostics_obj.get("failed_jobs")
    failed_jobs_list = failed_jobs if isinstance(failed_jobs, list) else []

    dispatch_command = payload.get("dispatch_command")
    dispatch_text = ""
    if isinstance(dispatch_command, list):
        dispatch_text = " ".join(str(item) for item in dispatch_command)

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
        payload = _read_payload(dispatch_path)
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
