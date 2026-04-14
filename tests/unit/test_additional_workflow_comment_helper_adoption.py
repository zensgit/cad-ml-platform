"""Regression checks for additional workflow PR comment helper adoption."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_workflow(name: str) -> dict:
    path = ROOT / ".github" / "workflows" / name
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_code_quality_workflow_uses_shared_helper_for_quality_report() -> None:
    workflow = _load_workflow("code-quality.yml")
    step = _get_step(workflow, "sonar-analysis", "Comment PR with quality report")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:code-quality-report -->'" in script
    assert "## 📊 Code Quality Report" in script
    assert "issueNumber: context.issue.number" in script
    assert "createComment(" not in script


def test_code_quality_workflow_uses_shared_helper_for_mypy_report() -> None:
    workflow = _load_workflow("code-quality.yml")
    step = _get_step(workflow, "type-check", "Comment PR with mypy issues")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:mypy-type-check-report -->'" in script
    assert "## 🔍 Type Check Report" in script
    assert "report.substring(0, 5000)" in script
    assert "createComment(" not in script


def test_security_audit_workflow_uses_shared_helper() -> None:
    workflow = _load_workflow("security-audit.yml")
    step = _get_step(workflow, "security-audit", "Comment on PR")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:security-audit-summary -->'" in script
    assert "body: `${marker}\\n${summary}`" in script
    assert "createComment(" not in script


def test_adaptive_rate_limit_monitor_workflow_uses_shared_helper() -> None:
    workflow = _load_workflow("adaptive-rate-limit-monitor.yml")
    steps = workflow["jobs"]["post-pr-comment"]["steps"]
    step = _get_step(workflow, "post-pr-comment", "Comment on PR")
    script = step["with"]["script"]

    assert any(
        str(item.get("uses", "")).startswith(
            "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
        )
        for item in steps
    )
    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "await upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:adaptive-rate-limit-pr-comment -->'" in script
    assert "body: `${marker}\\n${comment}`" in script
    assert "createComment(" not in script


def test_metrics_budget_check_workflow_uses_shared_helper() -> None:
    workflow = _load_workflow("metrics-budget-check.yml")
    step = _get_step(workflow, "post-pr-comment", "Create comment")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "await upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:metrics-budget-impact-analysis -->'" in script
    assert "body: `${marker}\\n${comment}`" in script
    assert "createComment(" not in script


def test_error_code_cleanup_workflow_uses_shared_comment_helper() -> None:
    workflow = _load_workflow("error-code-cleanup.yml")
    step = _get_step(workflow, "cleanup", "Add PR comment with details")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "upsertBotIssueComment({" in script
    assert "marker = '<!-- ci:error-code-cleanup-details -->'" in script
    assert "issueNumber: ${{ steps.pr.outputs.pull-request-number }}" in script
    assert "createComment(" not in script
