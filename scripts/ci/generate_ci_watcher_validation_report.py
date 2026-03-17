#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SUMMARY_GLOB = "watch_commit_*_summary.json"
SUMMARY_TOKEN_RE = re.compile(r"^watch_commit_(?P<token>.+)_summary$")
HEX_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _safe_int(value: object, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"failed to read json file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"failed to decode json file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid json payload in {path}: expected object")
    return payload


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to read text file {path}: {exc}") from exc


def _extract_summary_token(path: Path) -> str:
    match = SUMMARY_TOKEN_RE.match(path.stem)
    if not match:
        raise RuntimeError(
            "invalid summary file name; expected watch_commit_<sha>_summary.json: "
            f"{path.name}"
        )
    return match.group("token")


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


def _resolve_readiness_path(readiness_json: str, summary_path: Path) -> Path | None:
    explicit = str(readiness_json or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"readiness json does not exist: {path}")
        return path

    token = _extract_summary_token(summary_path)
    inferred = summary_path.parent / f"gh_readiness_watch_{token}.json"
    if inferred.exists():
        return inferred
    return None


def _resolve_soft_smoke_path(soft_smoke_json: str, summary_path: Path) -> Path | None:
    explicit = str(soft_smoke_json or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"soft smoke summary json does not exist: {path}")
        return path

    inferred = summary_path.parent / "evaluation_soft_mode_smoke_summary.json"
    if inferred.exists():
        return inferred
    return None


def _resolve_soft_smoke_md_path(
    soft_smoke_md: str,
    summary_path: Path,
    soft_smoke_path: Path | None,
) -> Path | None:
    explicit = str(soft_smoke_md or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"soft smoke summary md does not exist: {path}")
        return path

    if soft_smoke_path is not None:
        inferred = soft_smoke_path.with_suffix(".md")
        if inferred.exists():
            return inferred

    inferred = summary_path.parent / "evaluation_soft_mode_smoke_summary.md"
    if inferred.exists():
        return inferred
    return None


def _resolve_workflow_guardrail_summary_path(
    workflow_guardrail_summary_json: str,
    summary_path: Path,
) -> Path | None:
    explicit = str(workflow_guardrail_summary_json or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"workflow guardrail summary json does not exist: {path}")
        return path

    inferred = summary_path.parent / "workflow_guardrail_summary.json"
    if inferred.exists():
        return inferred
    return None


def _resolve_ci_workflow_guardrail_overview_path(
    ci_workflow_guardrail_overview_json: str,
    summary_path: Path,
) -> Path | None:
    explicit = str(ci_workflow_guardrail_overview_json or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise RuntimeError(f"ci workflow guardrail overview json does not exist: {path}")
        return path

    inferred = summary_path.parent / "ci_workflow_guardrail_overview.json"
    if inferred.exists():
        return inferred
    return None


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


def _resolve_output_path(
    *,
    output_md: str,
    report_dir: str,
    sha: str,
    report_sha_len: int,
    date_str: str,
    is_success: bool,
) -> Path:
    explicit = str(output_md or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    if report_sha_len < 1:
        raise RuntimeError("report-sha-len must be >= 1")

    sha_token = (sha[:report_sha_len] if sha else "UNKNOWN").upper()
    prefix = (
        "DEV_CI_WATCHER_SAFE_AUTO_SUCCESS_VALIDATION"
        if is_success
        else "DEV_CI_WATCHER_SAFE_AUTO_VALIDATION"
    )
    file_name = f"{prefix}_{sha_token}_{date_str}.md"
    return Path(report_dir).expanduser() / file_name


def _render_readiness_section(readiness: dict[str, Any] | None, readiness_path: Path | None) -> list[str]:
    if readiness is None or readiness_path is None:
        return [
            "## Readiness Artifact",
            "",
            "- Not found (inferred readiness json is missing).",
            "",
        ]

    lines = [
        "## Readiness Artifact",
        "",
        f"- `{readiness_path.as_posix()}`",
        f"- Result: `ok={bool(readiness.get('ok', False))}`",
    ]
    checks = readiness.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            ok = bool(item.get("ok", False))
            message = str(item.get("message") or "")
            lines.append(f"  - `{name}`: `ok={ok}` ({message})")
    lines.extend(["", ""])
    return lines


def _render_workflow_rows(summary: dict[str, Any]) -> list[str]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return []
    rows: list[str] = []
    for item in runs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("workflow_name") or "").strip()
        conclusion = str(item.get("conclusion") or "").strip() or "-"
        if name:
            rows.append(f"- {name}: {conclusion}")
    return rows


def _render_soft_smoke_section(
    soft_smoke: dict[str, Any] | None,
    soft_smoke_path: Path | None,
    soft_smoke_md_path: Path | None,
) -> list[str]:
    if soft_smoke is None or soft_smoke_path is None:
        return [
            "## Soft-Mode Smoke Artifact",
            "",
            "- Not found (inferred soft-smoke summary json is missing).",
            "",
        ]

    dispatch = soft_smoke.get("dispatch")
    if not isinstance(dispatch, dict):
        dispatch = {}
    attempts_raw = soft_smoke.get("attempts")
    attempts: list[dict[str, Any]] = []
    if isinstance(attempts_raw, list):
        for item in attempts_raw:
            if isinstance(item, dict):
                attempts.append(item)

    lines = [
        "## Soft-Mode Smoke Artifact",
        "",
        f"- `{soft_smoke_path.as_posix()}`",
        f"- `overall_exit_code={soft_smoke.get('overall_exit_code', 'n/a')}`",
        f"- `dispatch_exit_code={soft_smoke.get('dispatch_exit_code', 'n/a')}`",
        f"- `soft_marker_ok={soft_smoke.get('soft_marker_ok', 'n/a')}`",
        f"- `restore_ok={soft_smoke.get('restore_ok', 'n/a')}`",
        f"- `max_dispatch_attempts={soft_smoke.get('max_dispatch_attempts', 'n/a')}`",
        f"- `retry_sleep_seconds={soft_smoke.get('retry_sleep_seconds', 'n/a')}`",
        f"- `attempts_total={len(attempts)}`",
    ]
    run_id = dispatch.get("run_id")
    run_url = dispatch.get("run_url")
    if run_id:
        lines.append(f"- `run_id={run_id}`")
    if run_url:
        lines.append(f"- `run_url={run_url}`")
    if soft_smoke_md_path is not None:
        lines.append(f"- `rendered_markdown={soft_smoke_md_path.as_posix()}`")
    for idx, item in enumerate(attempts, start=1):
        attempt_no = item.get("attempt", idx)
        dispatch_exit = item.get("dispatch_exit_code", "n/a")
        marker_ok = item.get("soft_marker_ok", "n/a")
        message = item.get("soft_marker_message", item.get("message", "n/a"))
        lines.append(
            f"  - attempt#{attempt_no}: dispatch_exit_code={dispatch_exit}, "
            f"soft_marker_ok={marker_ok}, message={message}"
        )
    lines.extend(["", ""])
    return lines


def _render_workflow_guardrail_summary_section(
    workflow_guardrail_summary: dict[str, Any] | None,
    workflow_guardrail_summary_path: Path | None,
) -> list[str]:
    if workflow_guardrail_summary is None or workflow_guardrail_summary_path is None:
        return [
            "## Workflow Guardrail Summary",
            "",
            "- Not found (inferred workflow guardrail summary json is missing).",
            "",
        ]

    lines = [
        "## Workflow Guardrail Summary",
        "",
        f"- `{workflow_guardrail_summary_path.as_posix()}`",
        f"- `overall_status={workflow_guardrail_summary.get('overall_status', 'n/a')}`",
        f"- `overall_light={workflow_guardrail_summary.get('overall_light', 'n/a')}`",
        f"- `summary={workflow_guardrail_summary.get('summary', 'n/a')}`",
    ]
    for key in ("workflow_file_health", "workflow_inventory", "workflow_publish_helper"):
        payload = workflow_guardrail_summary.get(key)
        if not isinstance(payload, dict):
            continue
        status = payload.get("status", "n/a")
        summary = payload.get("summary", "n/a")
        lines.append(f"- `{key}.status={status}`")
        lines.append(f"- `{key}.summary={summary}`")
    lines.extend(["", ""])
    return lines


def _render_ci_workflow_guardrail_overview_section(
    ci_workflow_guardrail_overview: dict[str, Any] | None,
    ci_workflow_guardrail_overview_path: Path | None,
) -> list[str]:
    if ci_workflow_guardrail_overview is None or ci_workflow_guardrail_overview_path is None:
        return [
            "## CI Workflow Guardrail Overview",
            "",
            "- Not found (inferred ci workflow guardrail overview json is missing).",
            "",
        ]

    lines = [
        "## CI Workflow Guardrail Overview",
        "",
        f"- `{ci_workflow_guardrail_overview_path.as_posix()}`",
        f"- `overall_status={ci_workflow_guardrail_overview.get('overall_status', 'n/a')}`",
        f"- `overall_light={ci_workflow_guardrail_overview.get('overall_light', 'n/a')}`",
        f"- `summary={ci_workflow_guardrail_overview.get('summary', 'n/a')}`",
    ]
    for key in ("ci_watch", "workflow_guardrail"):
        payload = ci_workflow_guardrail_overview.get(key)
        if not isinstance(payload, dict):
            continue
        status = payload.get("status", "n/a")
        summary = payload.get("summary", "n/a")
        lines.append(f"- `{key}.status={status}`")
        lines.append(f"- `{key}.summary={summary}`")
    lines.extend(["", ""])
    return lines


def _render_failure_details_rows(summary: dict[str, Any]) -> list[str]:
    payload = summary.get("failure_details")
    if not isinstance(payload, list):
        return []
    rows: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        workflow_name = str(item.get("workflow_name") or "").strip() or "unknown"
        run_id = str(item.get("run_id") or "").strip() or "unknown"
        conclusion = str(item.get("conclusion") or "").strip() or "unknown"
        rows.append(f"- {workflow_name} (run={run_id}, conclusion={conclusion})")
        failed_jobs = item.get("failed_jobs")
        if isinstance(failed_jobs, list) and failed_jobs:
            rows.append(f"  - failed_jobs: {', '.join(str(v) for v in failed_jobs)}")
        failed_steps = item.get("failed_steps")
        if isinstance(failed_steps, list) and failed_steps:
            rows.append(f"  - failed_steps: {', '.join(str(v) for v in failed_steps)}")
        detail_unavailable = str(item.get("detail_unavailable") or "").strip()
        if detail_unavailable:
            rows.append(f"  - detail_unavailable: {detail_unavailable}")
        url = str(item.get("url") or "").strip()
        if url:
            rows.append(f"  - url: {url}")
    return rows


def _build_summary_payload(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    readiness: dict[str, Any] | None,
    readiness_path: Path | None,
    soft_smoke: dict[str, Any] | None,
    soft_smoke_path: Path | None,
    soft_smoke_md_path: Path | None,
    workflow_guardrail_summary: dict[str, Any] | None,
    workflow_guardrail_summary_path: Path | None,
    ci_workflow_guardrail_overview: dict[str, Any] | None,
    ci_workflow_guardrail_overview_path: Path | None,
    sha: str,
    date_str: str,
) -> dict[str, Any]:
    counts = _extract_counts(summary)
    verdict_success = _is_success_summary(summary)
    workflow_guardrail_status = "missing"
    workflow_guardrail_summary_text = ""
    if isinstance(workflow_guardrail_summary, dict):
        workflow_guardrail_status = str(
            workflow_guardrail_summary.get("overall_status") or "unknown"
        )
        workflow_guardrail_summary_text = str(
            workflow_guardrail_summary.get("summary") or ""
        )
    ci_workflow_overview_status = "missing"
    ci_workflow_overview_summary_text = ""
    if isinstance(ci_workflow_guardrail_overview, dict):
        ci_workflow_overview_status = str(
            ci_workflow_guardrail_overview.get("overall_status") or "unknown"
        )
        ci_workflow_overview_summary_text = str(
            ci_workflow_guardrail_overview.get("summary") or ""
        )
    report_summary = (
        f"verdict={'PASS' if verdict_success else 'FAIL'}, "
        f"reason={summary.get('reason', '')}, "
        f"failed={counts['failed']}, "
        f"missing_required={counts['missing_required']}, "
        f"workflow_guardrail={workflow_guardrail_status}, "
        f"ci_workflow_overview={ci_workflow_overview_status}"
    )
    attempts_total = 0
    if isinstance(soft_smoke, dict):
        attempts = soft_smoke.get("attempts")
        if isinstance(attempts, list):
            attempts_total = len([item for item in attempts if isinstance(item, dict)])
    return {
        "version": 1,
        "verdict": "PASS" if verdict_success else "FAIL",
        "verdict_success": verdict_success,
        "summary": report_summary,
        "date": date_str,
        "short_sha": sha[:7] if sha else "",
        "requested_sha": str(summary.get("requested_sha") or ""),
        "resolved_sha": str(summary.get("resolved_sha") or ""),
        "repo": str(summary.get("repo") or ""),
        "reason": str(summary.get("reason") or ""),
        "counts": counts,
        "summary_path": summary_path.as_posix(),
        "readiness_path": readiness_path.as_posix() if readiness_path is not None else "",
        "soft_smoke_path": soft_smoke_path.as_posix() if soft_smoke_path is not None else "",
        "soft_smoke_md_path": soft_smoke_md_path.as_posix()
        if soft_smoke_md_path is not None
        else "",
        "workflow_guardrail_summary_path": workflow_guardrail_summary_path.as_posix()
        if workflow_guardrail_summary_path is not None
        else "",
        "ci_workflow_guardrail_overview_path": ci_workflow_guardrail_overview_path.as_posix()
        if ci_workflow_guardrail_overview_path is not None
        else "",
        "sections": {
            "readiness": {
                "present": readiness is not None and readiness_path is not None,
                "ok": bool(readiness.get("ok")) if isinstance(readiness, dict) else None,
            },
            "soft_smoke": {
                "present": soft_smoke is not None and soft_smoke_path is not None,
                "overall_exit_code": soft_smoke.get("overall_exit_code")
                if isinstance(soft_smoke, dict)
                else None,
                "attempts_total": attempts_total,
            },
            "workflow_guardrail_summary": {
                "present": workflow_guardrail_summary is not None
                and workflow_guardrail_summary_path is not None,
                "overall_status": workflow_guardrail_status,
                "summary": workflow_guardrail_summary_text,
            },
            "ci_workflow_guardrail_overview": {
                "present": ci_workflow_guardrail_overview is not None
                and ci_workflow_guardrail_overview_path is not None,
                "overall_status": ci_workflow_overview_status,
                "summary": ci_workflow_overview_summary_text,
            },
        },
    }


def _render_markdown(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    readiness: dict[str, Any] | None,
    readiness_path: Path | None,
    soft_smoke: dict[str, Any] | None,
    soft_smoke_path: Path | None,
    soft_smoke_md_path: Path | None,
    workflow_guardrail_summary: dict[str, Any] | None,
    workflow_guardrail_summary_path: Path | None,
    ci_workflow_guardrail_overview: dict[str, Any] | None,
    ci_workflow_guardrail_overview_path: Path | None,
    sha: str,
    date_str: str,
) -> str:
    counts = _extract_counts(summary)
    verdict_success = _is_success_summary(summary)
    short_sha = (sha[:7] if sha else "unknown")
    success_conclusions = summary.get("success_conclusions")
    success_csv = "success,skipped,neutral"
    if isinstance(success_conclusions, list):
        normalized = [str(item).strip() for item in success_conclusions if str(item).strip()]
        if normalized:
            success_csv = ",".join(normalized)

    lines: list[str] = [
        f"# DEV CI Watcher Safe Auto Validation ({short_sha}, {date_str})",
        "",
        "## Objective",
        "",
        "Generate a standardized CI watcher validation report from watcher summary/readiness JSON artifacts.",
        "",
        "## Command",
        "",
        "```bash",
        "make watch-commit-workflows-safe-auto \\",
        f"  CI_WATCH_SHA={short_sha} \\",
        f"  CI_WATCH_REPO='{summary.get('repo', '')}' \\",
        "  CI_WATCH_ARTIFACT_SHA_LEN=12 \\",
        "  CI_WATCH_PRINT_FAILURE_DETAILS=1 \\",
        "  CI_WATCH_FAILURE_DETAILS_MAX_RUNS=5 \\",
        f"  CI_WATCH_SUCCESS_CONCLUSIONS='{success_csv}'",
        "```",
        "",
    ]
    lines.extend(_render_readiness_section(readiness, readiness_path))
    lines.extend(
        _render_soft_smoke_section(soft_smoke, soft_smoke_path, soft_smoke_md_path)
    )
    lines.extend(
        _render_workflow_guardrail_summary_section(
            workflow_guardrail_summary, workflow_guardrail_summary_path
        )
    )
    lines.extend(
        _render_ci_workflow_guardrail_overview_section(
            ci_workflow_guardrail_overview, ci_workflow_guardrail_overview_path
        )
    )
    lines.extend(
        [
            "## Watch Summary Artifact",
            "",
            f"- `{summary_path.as_posix()}`",
            f"- `requested_sha={summary.get('requested_sha', '')}`",
            f"- `resolved_sha={summary.get('resolved_sha', '')}`",
            f"- `repo={summary.get('repo', '')}`",
            f"- `exit_code={summary.get('exit_code', '')}`",
            f"- `reason={summary.get('reason', '')}`",
            f"- `counts.observed={counts['observed']}`",
            f"- `counts.completed={counts['completed']}`",
            f"- `counts.failed={counts['failed']}`",
            f"- `counts.missing_required={counts['missing_required']}`",
            f"- `duration_seconds={summary.get('duration_seconds', '')}`",
            "",
            "## Workflow Results",
            "",
        ]
    )
    workflow_rows = _render_workflow_rows(summary)
    if workflow_rows:
        lines.extend(workflow_rows)
    else:
        lines.append("- No workflow runs found in summary payload.")
    failure_rows = _render_failure_details_rows(summary)
    lines.extend(["", "## Failure Details", ""])
    if failure_rows:
        lines.extend(failure_rows)
    else:
        lines.append("- No structured failure_details in summary payload.")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "PASS." if verdict_success else "FAIL.",
            "",
            (
                "The commit satisfies release-gate CI criteria."
                if verdict_success
                else "The commit does not satisfy release-gate CI criteria."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CI watcher validation Markdown report from summary/readiness JSON artifacts."
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Explicit watcher summary json path. If empty, latest summary under --summary-dir is used.",
    )
    parser.add_argument(
        "--summary-dir",
        default="reports/ci",
        help="Directory containing watch_commit_*_summary.json files.",
    )
    parser.add_argument(
        "--readiness-json",
        default="",
        help="Optional explicit readiness json path. If empty, inferred from summary token.",
    )
    parser.add_argument(
        "--soft-smoke-summary-json",
        default="",
        help=(
            "Optional soft-mode smoke summary json path. If empty, "
            "infer <summary-dir>/evaluation_soft_mode_smoke_summary.json."
        ),
    )
    parser.add_argument(
        "--soft-smoke-summary-md",
        default="",
        help=(
            "Optional soft-mode smoke summary markdown path. If empty, "
            "infer sibling .md artifact when available."
        ),
    )
    parser.add_argument(
        "--workflow-guardrail-summary-json",
        default="",
        help=(
            "Optional workflow guardrail summary json path. If empty, "
            "infer <summary-dir>/workflow_guardrail_summary.json."
        ),
    )
    parser.add_argument(
        "--ci-workflow-guardrail-overview-json",
        default="",
        help=(
            "Optional CI workflow guardrail overview json path. If empty, "
            "infer <summary-dir>/ci_workflow_guardrail_overview.json."
        ),
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional explicit markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional explicit JSON summary output path.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Report directory when --output-md is not set.",
    )
    parser.add_argument(
        "--report-sha-len",
        type=int,
        default=7,
        help="SHA prefix length in generated report filename when --output-md is not set.",
    )
    parser.add_argument(
        "--date",
        default="",
        help="Date token used in output filename/title (YYYYMMDD). Defaults to today.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary_path = _resolve_summary_path(str(args.summary_json), str(args.summary_dir))
    summary = _read_json_dict(summary_path)

    readiness_path = _resolve_readiness_path(str(args.readiness_json), summary_path)
    readiness: dict[str, Any] | None = None
    if readiness_path is not None:
        readiness = _read_json_dict(readiness_path)
    soft_smoke_path = _resolve_soft_smoke_path(
        str(args.soft_smoke_summary_json), summary_path
    )
    soft_smoke: dict[str, Any] | None = None
    if soft_smoke_path is not None:
        soft_smoke = _read_json_dict(soft_smoke_path)
    soft_smoke_md_path = _resolve_soft_smoke_md_path(
        str(args.soft_smoke_summary_md), summary_path, soft_smoke_path
    )
    if soft_smoke_md_path is not None:
        _read_text(soft_smoke_md_path)
    workflow_guardrail_summary_path = _resolve_workflow_guardrail_summary_path(
        str(args.workflow_guardrail_summary_json), summary_path
    )
    workflow_guardrail_summary: dict[str, Any] | None = None
    if workflow_guardrail_summary_path is not None:
        workflow_guardrail_summary = _read_json_dict(workflow_guardrail_summary_path)
    ci_workflow_guardrail_overview_path = _resolve_ci_workflow_guardrail_overview_path(
        str(args.ci_workflow_guardrail_overview_json), summary_path
    )
    ci_workflow_guardrail_overview: dict[str, Any] | None = None
    if ci_workflow_guardrail_overview_path is not None:
        ci_workflow_guardrail_overview = _read_json_dict(
            ci_workflow_guardrail_overview_path
        )

    sha = _extract_sha(summary)
    date_str = str(args.date or "").strip() or datetime.now().strftime("%Y%m%d")
    is_success = _is_success_summary(summary)
    output_path = _resolve_output_path(
        output_md=str(args.output_md),
        report_dir=str(args.report_dir),
        sha=sha,
        report_sha_len=int(args.report_sha_len),
        date_str=date_str,
        is_success=is_success,
    )
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = _render_markdown(
        summary=summary,
        summary_path=summary_path,
        readiness=readiness,
        readiness_path=readiness_path,
        soft_smoke=soft_smoke,
        soft_smoke_path=soft_smoke_path,
        soft_smoke_md_path=soft_smoke_md_path,
        workflow_guardrail_summary=workflow_guardrail_summary,
        workflow_guardrail_summary_path=workflow_guardrail_summary_path,
        ci_workflow_guardrail_overview=ci_workflow_guardrail_overview,
        ci_workflow_guardrail_overview_path=ci_workflow_guardrail_overview_path,
        sha=sha,
        date_str=date_str,
    )
    summary_payload = _build_summary_payload(
        summary=summary,
        summary_path=summary_path,
        readiness=readiness,
        readiness_path=readiness_path,
        soft_smoke=soft_smoke,
        soft_smoke_path=soft_smoke_path,
        soft_smoke_md_path=soft_smoke_md_path,
        workflow_guardrail_summary=workflow_guardrail_summary,
        workflow_guardrail_summary_path=workflow_guardrail_summary_path,
        ci_workflow_guardrail_overview=ci_workflow_guardrail_overview,
        ci_workflow_guardrail_overview_path=ci_workflow_guardrail_overview_path,
        sha=sha,
        date_str=date_str,
    )
    output_path.write_text(markdown, encoding="utf-8")
    output_json = str(args.output_json or "").strip()
    if output_json:
        output_json_path = Path(output_json).expanduser()
        if output_json_path.parent != Path("."):
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output_json={output_json_path.as_posix()}", flush=True)

    print(f"summary_json={summary_path.as_posix()}", flush=True)
    print(
        "readiness_json="
        + (readiness_path.as_posix() if readiness_path is not None else "(not found)"),
        flush=True,
    )
    print(
        "soft_smoke_json="
        + (soft_smoke_path.as_posix() if soft_smoke_path is not None else "(not found)"),
        flush=True,
    )
    print(
        "soft_smoke_md="
        + (
            soft_smoke_md_path.as_posix()
            if soft_smoke_md_path is not None
            else "(not found)"
        ),
        flush=True,
    )
    print(f"output_md={output_path.as_posix()}", flush=True)
    print(f"verdict={'PASS' if is_success else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
