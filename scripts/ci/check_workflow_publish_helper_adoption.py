#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COMMENT_HELPER_REQUIRED = {
    "code-quality.yml",
    "error-code-cleanup.yml",
    "metrics-budget-check.yml",
    "pr-auto-label-comment.yml",
    "release-risk-check.yml",
    "sbom.yml",
    "security-audit.yml",
}

ISSUE_HELPER_REQUIRED = {
    "adaptive-rate-limit-monitor.yml",
    "badge-review.yml",
}

RAW_PUBLISH_PATTERNS = {
    "raw_create_comment": "github.rest.issues.createComment(",
    "raw_update_comment": "github.rest.issues.updateComment(",
    "raw_list_comments": "github.rest.issues.listComments(",
    "raw_create_issue": "github.rest.issues.create(",
    "raw_update_issue": "github.rest.issues.update(",
    "raw_list_issues": "github.rest.issues.listForRepo(",
}

COMMENT_HELPER_IMPORT = "require('./scripts/ci/comment_pr_utils.js')"
ISSUE_HELPER_IMPORT = "require('./scripts/ci/issue_upsert_utils.js')"


def build_report(workflow_root: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for path in sorted(workflow_root.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        issues: list[str] = []

        found_patterns = [
            key for key, token in RAW_PUBLISH_PATTERNS.items() if token in text
        ]
        if found_patterns:
            issues.append(
                "raw GitHub publish calls found: " + ", ".join(sorted(found_patterns))
            )

        if path.name in COMMENT_HELPER_REQUIRED and COMMENT_HELPER_IMPORT not in text:
            issues.append("missing shared comment helper import")

        if path.name in ISSUE_HELPER_REQUIRED and ISSUE_HELPER_IMPORT not in text:
            issues.append("missing shared issue helper import")

        results.append(
            {
                "filename": path.name,
                "requires_comment_helper": path.name in COMMENT_HELPER_REQUIRED,
                "requires_issue_helper": path.name in ISSUE_HELPER_REQUIRED,
                "has_comment_helper_import": COMMENT_HELPER_IMPORT in text,
                "has_issue_helper_import": ISSUE_HELPER_IMPORT in text,
                "raw_publish_patterns": found_patterns,
                "ok": not issues,
                "issues": issues,
            }
        )

    failed = [row for row in results if not row["ok"]]
    raw_publish_violation_count = sum(
        1 for row in results if row.get("raw_publish_patterns")
    )
    missing_comment_helper_import_count = sum(
        1
        for row in results
        if row.get("requires_comment_helper") and not row.get("has_comment_helper_import")
    )
    missing_issue_helper_import_count = sum(
        1
        for row in results
        if row.get("requires_issue_helper") and not row.get("has_issue_helper_import")
    )
    return {
        "version": 1,
        "workflow_root": str(workflow_root),
        "checked_count": len(results),
        "failed_count": len(failed),
        "raw_publish_violation_count": raw_publish_violation_count,
        "missing_comment_helper_import_count": missing_comment_helper_import_count,
        "missing_issue_helper_import_count": missing_issue_helper_import_count,
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    failing_rows = [row for row in report.get("results", []) if not row.get("ok")]
    lines = [
        "# Workflow Publish Helper Adoption",
        "",
        f"- workflow_root: `{report.get('workflow_root')}`",
        f"- checked_count: `{report.get('checked_count')}`",
        f"- failed_count: `{report.get('failed_count')}`",
        f"- raw_publish_violation_count: `{report.get('raw_publish_violation_count')}`",
        f"- missing_comment_helper_import_count: `{report.get('missing_comment_helper_import_count')}`",
        f"- missing_issue_helper_import_count: `{report.get('missing_issue_helper_import_count')}`",
        "",
        "## Issue Summary",
        "",
    ]
    if failing_rows:
        for row in failing_rows:
            issues = ", ".join(row.get("issues") or []) or "unknown"
            lines.append(f"- {row.get('filename')}: {issues}")
    else:
        lines.append("- none")

    lines.extend(["", "## Workflow Files", ""])
    for row in report.get("results", []):
        raw_patterns = ",".join(row.get("raw_publish_patterns") or []) or "(none)"
        lines.append(
            "- "
            f"{row.get('filename')}: "
            f"ok={row.get('ok')} "
            f"requires_comment_helper={row.get('requires_comment_helper')} "
            f"requires_issue_helper={row.get('requires_issue_helper')} "
            f"raw_publish_patterns={raw_patterns}"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that workflow-side GitHub publish logic reuses shared helpers."
    )
    parser.add_argument(
        "--workflow-root",
        default=".github/workflows",
        help="Workflow directory root (default: .github/workflows).",
    )
    parser.add_argument(
        "--summary-json-out",
        default="",
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown summary output path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    workflow_root = Path(args.workflow_root).expanduser()
    report = build_report(workflow_root)

    if args.summary_json_out:
        out = Path(args.summary_json_out).expanduser()
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.output_md:
        out = Path(args.output_md).expanduser()
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")

    if report["failed_count"]:
        print(
            f"found {report['failed_count']} workflow(s) with raw publish logic or missing helper imports"
        )
        return 1

    print(f"ok: workflow publish helper adoption passed for {report['checked_count']} workflow(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
