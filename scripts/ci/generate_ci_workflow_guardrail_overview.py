#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SUMMARY_GLOB = "watch_commit_*_summary.json"
SUMMARY_TOKEN_RE = re.compile(r"^watch_commit_(?P<token>.+)_summary$")
HEX_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _safe_int(value: object, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_summary_path(summary_json: str, summary_dir: str) -> Path:
    explicit = str(summary_json or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"summary json does not exist: {path}")
        return path

    directory = Path(summary_dir).expanduser()
    candidates = sorted(
        directory.glob(SUMMARY_GLOB),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"no summary json found in {directory} matching pattern {SUMMARY_GLOB}"
        )
    return candidates[0]


def _extract_sha(summary: dict[str, Any]) -> str:
    for key in ("resolved_sha", "requested_sha"):
        raw = str(summary.get(key) or "").strip()
        if raw and HEX_SHA_RE.match(raw):
            return raw.lower()
    return ""


def _extract_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts_payload = summary.get("counts")
    if not isinstance(counts_payload, dict):
        counts_payload = {}

    runs_payload = summary.get("runs")
    runs_list: list[dict[str, Any]] = []
    if isinstance(runs_payload, list):
        for item in runs_payload:
            if isinstance(item, dict):
                runs_list.append(item)

    observed = int(counts_payload.get("observed", len(runs_list)) or 0)
    completed = int(
        counts_payload.get(
            "completed",
            sum(1 for item in runs_list if str(item.get("status") or "") == "completed"),
        )
        or 0
    )
    failed = int(
        counts_payload.get(
            "failed",
            len(summary.get("failed_workflows") or []),
        )
        or 0
    )
    missing_required = int(
        counts_payload.get(
            "missing_required",
            len(summary.get("missing_required") or []),
        )
        or 0
    )

    return {
        "observed": observed,
        "completed": completed,
        "failed": failed,
        "missing_required": missing_required,
    }


def _is_success_summary(summary: dict[str, Any]) -> bool:
    counts = _extract_counts(summary)
    return (
        _safe_int(summary.get("exit_code"), 1) == 0
        and str(summary.get("reason") or "") == "all_workflows_success"
        and counts["failed"] == 0
        and counts["missing_required"] == 0
    )


def _read_json_dict(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid json payload in {path}: expected object")
    return payload


def _build_ci_watch_section(payload: dict[str, Any]) -> dict[str, Any]:
    counts = _extract_counts(payload)
    reason = str(payload.get("reason") or "unknown")
    resolved_sha = str(payload.get("resolved_sha") or payload.get("requested_sha") or "").strip()
    exit_code = int(payload.get("exit_code", 1) or 0)
    success = _is_success_summary(payload)
    summary = (
        f"reason={reason}, observed={counts['observed']}, completed={counts['completed']}, "
        f"failed={counts['failed']}, missing_required={counts['missing_required']}"
    )
    return {
        "status": "ok" if success else "error",
        "light": "🟢" if success else "🔴",
        "summary": summary,
        "reason": reason,
        "resolved_sha": resolved_sha,
        "exit_code": exit_code,
        "counts": counts,
    }


def _build_workflow_guardrail_section(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("overall_status") or "unknown")
    light = str(payload.get("overall_light") or "").strip()
    summary = str(payload.get("summary") or f"status={status}")
    if not light:
        if status == "ok":
            light = "🟢"
        elif status == "warning":
            light = "🟡"
        else:
            light = "🔴"
    return {
        "status": status,
        "light": light,
        "summary": summary,
        "workflow_file_health": payload.get("workflow_file_health"),
        "workflow_inventory": payload.get("workflow_inventory"),
        "workflow_publish_helper": payload.get("workflow_publish_helper"),
    }


def build_report(
    *,
    ci_watch_payload: dict[str, Any],
    workflow_guardrail_payload: dict[str, Any],
) -> dict[str, Any]:
    ci_watch = _build_ci_watch_section(ci_watch_payload)
    workflow_guardrail = _build_workflow_guardrail_section(workflow_guardrail_payload)
    statuses = [ci_watch["status"], workflow_guardrail["status"]]
    if "error" in statuses:
        overall_status = "error"
        overall_light = "🔴"
    elif "warning" in statuses:
        overall_status = "warning"
        overall_light = "🟡"
    else:
        overall_status = "ok"
        overall_light = "🟢"
    summary = (
        f"status={overall_status}, ci_watch={ci_watch['status']}, "
        f"workflow_guardrail={workflow_guardrail['status']}"
    )
    return {
        "version": 1,
        "overall_status": overall_status,
        "overall_light": overall_light,
        "summary": summary,
        "sha": _extract_sha(ci_watch_payload),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ci_watch": ci_watch,
        "workflow_guardrail": workflow_guardrail,
    }


def render_markdown(report: dict[str, Any]) -> str:
    ci_watch = report["ci_watch"]
    workflow_guardrail = report["workflow_guardrail"]
    lines = [
        "# CI Workflow Guardrail Overview",
        "",
        f"- overall_status: `{report.get('overall_status')}`",
        f"- overall_light: `{report.get('overall_light')}`",
        f"- summary: `{report.get('summary')}`",
        f"- sha: `{report.get('sha') or 'unknown'}`",
        "",
        "## CI Watch",
        "",
        f"- status: `{ci_watch['status']}` {ci_watch['light']}",
        f"- summary: `{ci_watch['summary']}`",
        f"- reason: `{ci_watch['reason']}`",
        f"- resolved_sha: `{ci_watch['resolved_sha'] or 'unknown'}`",
        "",
        "## Workflow Guardrail",
        "",
        f"- status: `{workflow_guardrail['status']}` {workflow_guardrail['light']}",
        f"- summary: `{workflow_guardrail['summary']}`",
        "",
        "## Release Gate Readiness",
        "",
        (
            "- PASS: watcher and workflow guardrails are both green."
            if report.get("overall_status") == "ok"
            else "- FAIL: at least one of watcher or workflow guardrails is degraded."
        ),
        "",
    ]
    return "\n".join(lines)


def _write_text(path_value: str, text: str) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate CI watcher summary and workflow guardrail summary."
    )
    parser.add_argument(
        "--ci-watch-summary-json",
        default="",
        help="Explicit watcher summary json path. If empty, latest summary under --ci-watch-summary-dir is used.",
    )
    parser.add_argument(
        "--ci-watch-summary-dir",
        default="reports/ci",
        help="Directory containing watch_commit_*_summary.json files.",
    )
    parser.add_argument("--workflow-guardrail-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ci_watch_summary_path = _resolve_summary_path(
        str(args.ci_watch_summary_json), str(args.ci_watch_summary_dir)
    )
    report = build_report(
        ci_watch_payload=_read_json_dict(ci_watch_summary_path.as_posix()),
        workflow_guardrail_payload=_read_json_dict(args.workflow_guardrail_json),
    )
    _write_text(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_text(args.output_md, render_markdown(report) + "\n")
    print(f"ci_watch_summary_json={ci_watch_summary_path.as_posix()}")
    print(f"output_json={args.output_json}")
    print(f"output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
