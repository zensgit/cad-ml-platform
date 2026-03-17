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
    return {
        "version": 1,
        "workflow_root": str(workflow_root),
        "checked_count": len(results),
        "failed_count": len(failed),
        "results": results,
    }


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

    if report["failed_count"]:
        print(
            f"found {report['failed_count']} workflow(s) with raw publish logic or missing helper imports"
        )
        return 1

    print(f"ok: workflow publish helper adoption passed for {report['checked_count']} workflow(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
