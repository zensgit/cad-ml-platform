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


def _render_markdown(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    readiness: dict[str, Any] | None,
    readiness_path: Path | None,
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
        "  CI_WATCH_ARTIFACT_SHA_LEN=12 \\",
        f"  CI_WATCH_SUCCESS_CONCLUSIONS='{success_csv}'",
        "```",
        "",
    ]
    lines.extend(_render_readiness_section(readiness, readiness_path))
    lines.extend(
        [
            "## Watch Summary Artifact",
            "",
            f"- `{summary_path.as_posix()}`",
            f"- `requested_sha={summary.get('requested_sha', '')}`",
            f"- `resolved_sha={summary.get('resolved_sha', '')}`",
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
        "--output-md",
        default="",
        help="Optional explicit markdown output path.",
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
        sha=sha,
        date_str=date_str,
    )
    output_path.write_text(markdown, encoding="utf-8")

    print(f"summary_json={summary_path.as_posix()}", flush=True)
    print(
        "readiness_json="
        + (readiness_path.as_posix() if readiness_path is not None else "(not found)"),
        flush=True,
    )
    print(f"output_md={output_path.as_posix()}", flush=True)
    print(f"verdict={'PASS' if is_success else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
