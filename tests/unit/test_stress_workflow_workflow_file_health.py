"""Regression checks for workflow-file-health wiring in stress-tests workflow."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "stress-tests.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_push_paths_include_workflow_health_inputs() -> None:
    workflow = _load_workflow()
    push_paths = workflow["on"]["push"]["paths"]
    assert "scripts/ci/check_workflow_file_issues.py" in push_paths
    assert ".github/workflows/*.yml" in push_paths


def test_stress_workflow_has_workflow_file_health_job() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["workflow-file-health"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"

    step = _get_step(workflow, "workflow-file-health", "Validate workflow file health via GitHub parser")
    run_script = step["run"]
    assert "scripts/ci/check_workflow_file_issues.py" in run_script
    assert "--mode gh" in run_script
    assert '--ref "${GITHUB_SHA}"' in run_script
    assert "--summary-json-out reports/ci/workflow_file_health_summary.json" in run_script

    upload = _get_step(workflow, "workflow-file-health", "Upload workflow health summary")
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "workflow-file-health-${{ github.run_number }}"
    assert upload["with"]["path"] == "reports/ci/workflow_file_health_summary.json"


def test_stress_jobs_depend_on_workflow_file_health() -> None:
    workflow = _load_workflow()
    assert workflow["jobs"]["metrics-consistency"]["needs"] == "workflow-file-health"
    assert workflow["jobs"]["stress-unit-tests"]["needs"] == "workflow-file-health"
