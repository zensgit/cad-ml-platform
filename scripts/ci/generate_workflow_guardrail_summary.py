#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json_dict(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid json payload in {path}: expected object")
    return payload


def _build_workflow_file_health_section(payload: dict[str, Any]) -> dict[str, Any]:
    failed_count = int(payload.get("failed_count", 0) or 0)
    count = int(payload.get("count", 0) or 0)
    mode_used = str(payload.get("mode_used") or "unknown")
    fallback_reason = str(payload.get("fallback_reason") or "none")
    summary = f"failed={failed_count}/{count}, mode={mode_used}, fallback={fallback_reason}"
    return {
        "status": "error" if failed_count > 0 else "ok",
        "light": "🔴" if failed_count > 0 else "🟢",
        "summary": summary,
        "count": count,
        "failed_count": failed_count,
        "mode_used": mode_used,
        "fallback_reason": fallback_reason,
    }


def _build_workflow_inventory_section(payload: dict[str, Any]) -> dict[str, Any]:
    workflow_count = int(payload.get("workflow_count", 0) or 0)
    duplicate_count = int(payload.get("duplicate_name_count", 0) or 0)
    missing_required_count = int(payload.get("missing_required_count", 0) or 0)
    non_unique_required_count = int(payload.get("non_unique_required_count", 0) or 0)
    summary = (
        f"workflows={workflow_count}, duplicate={duplicate_count}, "
        f"missing_required={missing_required_count}, non_unique_required={non_unique_required_count}"
    )
    bad = duplicate_count > 0 or missing_required_count > 0 or non_unique_required_count > 0
    return {
        "status": "error" if bad else "ok",
        "light": "🔴" if bad else "🟢",
        "summary": summary,
        "workflow_count": workflow_count,
        "duplicate_name_count": duplicate_count,
        "missing_required_count": missing_required_count,
        "non_unique_required_count": non_unique_required_count,
    }


def _build_workflow_publish_helper_section(payload: dict[str, Any]) -> dict[str, Any]:
    checked_count = int(payload.get("checked_count", 0) or 0)
    failed_count = int(payload.get("failed_count", 0) or 0)
    raw_publish_violation_count = int(payload.get("raw_publish_violation_count", 0) or 0)
    missing_comment_helper_import_count = int(
        payload.get("missing_comment_helper_import_count", 0) or 0
    )
    missing_issue_helper_import_count = int(
        payload.get("missing_issue_helper_import_count", 0) or 0
    )
    summary = (
        f"checked={checked_count}, failed={failed_count}, raw={raw_publish_violation_count}, "
        f"missing_comment_helper={missing_comment_helper_import_count}, "
        f"missing_issue_helper={missing_issue_helper_import_count}"
    )
    bad = (
        failed_count > 0
        or raw_publish_violation_count > 0
        or missing_comment_helper_import_count > 0
        or missing_issue_helper_import_count > 0
    )
    return {
        "status": "error" if bad else "ok",
        "light": "🔴" if bad else "🟢",
        "summary": summary,
        "checked_count": checked_count,
        "failed_count": failed_count,
        "raw_publish_violation_count": raw_publish_violation_count,
        "missing_comment_helper_import_count": missing_comment_helper_import_count,
        "missing_issue_helper_import_count": missing_issue_helper_import_count,
    }


def build_report(
    *,
    workflow_file_health_payload: dict[str, Any],
    workflow_inventory_payload: dict[str, Any],
    workflow_publish_helper_payload: dict[str, Any],
) -> dict[str, Any]:
    workflow_file_health = _build_workflow_file_health_section(workflow_file_health_payload)
    workflow_inventory = _build_workflow_inventory_section(workflow_inventory_payload)
    workflow_publish_helper = _build_workflow_publish_helper_section(
        workflow_publish_helper_payload
    )
    sections = [
        workflow_file_health["status"],
        workflow_inventory["status"],
        workflow_publish_helper["status"],
    ]
    error_count = sum(1 for item in sections if item == "error")
    warn_count = sum(1 for item in sections if item == "warning")
    if error_count > 0:
        overall_status = "error"
        overall_light = "🔴"
    elif warn_count > 0:
        overall_status = "warning"
        overall_light = "🟡"
    else:
        overall_status = "ok"
        overall_light = "🟢"
    summary = (
        f"status={overall_status}, "
        f"workflow_health={workflow_file_health['status']}, "
        f"inventory={workflow_inventory['status']}, "
        f"publish_helper={workflow_publish_helper['status']}"
    )
    return {
        "version": 1,
        "overall_status": overall_status,
        "overall_light": overall_light,
        "summary": summary,
        "workflow_file_health": workflow_file_health,
        "workflow_inventory": workflow_inventory,
        "workflow_publish_helper": workflow_publish_helper,
    }


def render_markdown(report: dict[str, Any]) -> str:
    workflow_file_health = report["workflow_file_health"]
    workflow_inventory = report["workflow_inventory"]
    workflow_publish_helper = report["workflow_publish_helper"]
    lines = [
        "# Workflow Guardrail Summary",
        "",
        f"- overall_status: `{report.get('overall_status')}`",
        f"- overall_light: `{report.get('overall_light')}`",
        f"- summary: `{report.get('summary')}`",
        "",
        "## Sections",
        "",
        f"- workflow_file_health: {workflow_file_health['light']} {workflow_file_health['summary']}",
        f"- workflow_inventory: {workflow_inventory['light']} {workflow_inventory['summary']}",
        f"- workflow_publish_helper: {workflow_publish_helper['light']} {workflow_publish_helper['summary']}",
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
        description="Aggregate workflow health/inventory/publish-helper summaries."
    )
    parser.add_argument("--workflow-file-health-json", required=True)
    parser.add_argument("--workflow-inventory-json", required=True)
    parser.add_argument("--workflow-publish-helper-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_report(
        workflow_file_health_payload=_read_json_dict(args.workflow_file_health_json),
        workflow_inventory_payload=_read_json_dict(args.workflow_inventory_json),
        workflow_publish_helper_payload=_read_json_dict(args.workflow_publish_helper_json),
    )
    _write_text(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_text(args.output_md, render_markdown(report) + "\n")
    print(f"output_json={args.output_json}")
    print(f"output_md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
