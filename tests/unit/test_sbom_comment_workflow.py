"""Regression checks for sbom workflow PR comment helper adoption."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "sbom.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_sbom_workflow_uses_shared_pr_comment_helper() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "sbom-diff", "Comment PR with diff")
    script = step["with"]["script"]

    assert "require('./scripts/ci/comment_pr_utils.js')" in script
    assert "upsertBotIssueComment({" in script
    assert "issueNumber: context.issue.number" in script
    assert "marker: 'SBOM Dependency Changes'" in script
    assert "body," in script
    assert "listComments(" not in script
    assert "updateComment(" not in script
    assert "createComment(" not in script
