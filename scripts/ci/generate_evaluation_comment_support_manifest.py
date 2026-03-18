#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORT_ARTIFACTS: list[tuple[str, str, str]] = [
    ("workflow_file_health_json", "workflow_file_health_for_comment.json", "json"),
    ("workflow_inventory_json", "workflow_inventory_for_comment.json", "json"),
    ("workflow_inventory_md", "workflow_inventory_for_comment.md", "md"),
    ("workflow_publish_helper_json", "workflow_publish_helper_for_comment.json", "json"),
    ("workflow_publish_helper_md", "workflow_publish_helper_for_comment.md", "md"),
    ("workflow_guardrail_json", "workflow_guardrail_for_comment.json", "json"),
    ("workflow_guardrail_md", "workflow_guardrail_for_comment.md", "md"),
    (
        "ci_workflow_guardrail_overview_json",
        "ci_workflow_guardrail_overview_for_comment.json",
        "json",
    ),
    (
        "ci_workflow_guardrail_overview_md",
        "ci_workflow_guardrail_overview_for_comment.md",
        "md",
    ),
    ("ci_watch_validation_json", "ci_watch_validation_for_comment.json", "json"),
    ("ci_watch_validation_md", "ci_watch_validation_for_comment.md", "md"),
]


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid json payload in {path}: expected object")
    return payload


def _extract_summary(payload: dict[str, Any]) -> str:
    summary = str(payload.get("summary") or "").strip()
    if summary:
        return summary
    verdict = str(payload.get("verdict") or "").strip()
    overall_status = str(payload.get("overall_status") or "").strip()
    if verdict:
        return f"verdict={verdict}"
    if overall_status:
        return f"status={overall_status}"
    return "n/a"


def build_report(*, reports_dir: str) -> dict[str, Any]:
    root = Path(reports_dir).expanduser()
    entries: list[dict[str, Any]] = []
    present_count = 0
    missing_count = 0
    invalid_count = 0

    for artifact_id, rel_path, kind in SUPPORT_ARTIFACTS:
        path = root / rel_path
        entry: dict[str, Any] = {
            "id": artifact_id,
            "path": rel_path,
            "kind": kind,
            "present": path.exists(),
        }
        if path.exists():
            present_count += 1
            if kind == "json":
                try:
                    payload = _read_json_dict(path)
                    entry["parse_status"] = "ok"
                    entry["summary"] = _extract_summary(payload)
                except Exception as exc:  # pragma: no cover - exercised via tests
                    invalid_count += 1
                    entry["parse_status"] = "error"
                    entry["summary"] = f"parse_error: {exc}"
            else:
                entry["summary"] = "present"
        else:
            missing_count += 1
            entry["summary"] = "missing"
        entries.append(entry)

    if invalid_count > 0:
        overall_status = "error"
        overall_light = "🔴"
    elif missing_count > 0:
        overall_status = "warning"
        overall_light = "🟡"
    else:
        overall_status = "ok"
        overall_light = "🟢"

    summary = (
        f"present={present_count}/{len(entries)}, missing={missing_count}, invalid={invalid_count}"
    )
    return {
        "version": 1,
        "overall_status": overall_status,
        "overall_light": overall_light,
        "summary": summary,
        "reports_dir": root.as_posix(),
        "present_count": present_count,
        "missing_count": missing_count,
        "invalid_count": invalid_count,
        "entries": entries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Comment Support Manifest",
        "",
        f"- overall_status: `{report.get('overall_status')}`",
        f"- overall_light: `{report.get('overall_light')}`",
        f"- summary: `{report.get('summary')}`",
        f"- reports_dir: `{report.get('reports_dir')}`",
        "",
        "## Entries",
        "",
    ]
    for entry in report.get("entries", []):
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"- `{entry.get('id')}` [{entry.get('kind')}] `{entry.get('path')}`: "
            f"{'present' if entry.get('present') else 'missing'} :: "
            f"`{entry.get('summary', 'n/a')}`"
        )
    lines.extend(["", "## Verdict", ""])
    if str(report.get("overall_status")) == "ok":
        lines.append("- PASS: evaluation comment support bundle is complete.")
    elif str(report.get("overall_status")) == "warning":
        lines.append("- WARN: evaluation comment support bundle is partial.")
    else:
        lines.append("- FAIL: evaluation comment support bundle contains invalid JSON.")
    lines.append("")
    return "\n".join(lines)


def _write_text(path_value: str, text: str) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manifest for evaluation-report comment support artifacts."
    )
    parser.add_argument("--reports-dir", default="reports/ci")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_report(reports_dir=str(args.reports_dir))
    _write_text(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_text(args.output_md, render_markdown(report) + "\n")
    print(f"output_json={args.output_json}")
    print(f"output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
