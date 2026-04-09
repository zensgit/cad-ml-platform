"""Regression checks for workflow issue helper adoption."""

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


def test_badge_review_workflow_uses_shared_issue_helper() -> None:
    workflow = _load_workflow("badge-review.yml")
    step = _get_step(workflow, "review-badges", "Create review issue")
    script = step["with"]["script"]

    assert "require('./scripts/ci/issue_upsert_utils.js')" in script
    assert "upsertOpenIssue({" in script
    assert "marker = `<!-- ci:badge-review:${year}-${month} -->`" in script
    assert "listLabels: 'badge-review'" in script
    assert "title = `🎯 Monthly Badge Review - ${month} ${year}`" in script
    assert "issues.listForRepo(" not in script
    assert "issues.create(" not in script


def test_adaptive_rate_limit_monitor_workflow_uses_shared_issue_helper() -> None:
    workflow = _load_workflow("adaptive-rate-limit-monitor.yml")
    step = _get_step(workflow, "notify-alerts", "Create GitHub issue")
    script = step["with"]["script"]

    assert "require('./scripts/ci/issue_upsert_utils.js')" in script
    assert "upsertOpenIssue({" in script
    assert "marker = '<!-- ci:adaptive-rate-limit-alert -->'" in script
    assert "listLabels: 'alert,performance,rate-limiting'" in script
    assert "issues.create(" not in script
