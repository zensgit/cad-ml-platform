"""Regression checks for evaluation soft-mode smoke workflow wiring."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "evaluation-soft-mode-smoke.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_workflow_has_schedule_and_dispatch_inputs() -> None:
    workflow = _load_workflow()
    assert "workflow_dispatch" in workflow["on"]
    assert "schedule" in workflow["on"]
    schedule = workflow["on"]["schedule"]
    assert isinstance(schedule, list) and schedule
    assert schedule[0]["cron"] == "20 3 * * *"

    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "ref" in inputs
    assert "expected_conclusion" in inputs
    assert "keep_soft" in inputs
    assert "skip_log_check" in inputs


def test_workflow_job_permissions_and_dispatch_step_wiring() -> None:
    workflow = _load_workflow()
    permissions = workflow["permissions"]
    assert permissions["contents"] == "read"
    assert permissions["actions"] == "write"

    job = workflow["jobs"]["soft-mode-smoke"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"

    run_step = _get_step(
        workflow, "soft-mode-smoke", "Run evaluation soft-mode smoke dispatcher"
    )
    run_script = run_step["run"]
    assert "scripts/ci/dispatch_evaluation_soft_mode_smoke.py" in run_script
    assert '--workflow "evaluation-report.yml"' in run_script
    assert "--ref \"$REF_INPUT\"" in run_script
    assert "--expected-conclusion \"$EXPECTED_INPUT\"" in run_script
    assert "--output-json reports/ci/evaluation_soft_mode_smoke_summary.json" in run_script

    upload_step = _get_step(
        workflow, "soft-mode-smoke", "Upload soft-mode smoke summary"
    )
    assert (
        upload_step["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert (
        upload_step["with"]["name"]
        == "evaluation-soft-mode-smoke-${{ github.run_number }}"
    )
    assert (
        upload_step["with"]["path"]
        == "reports/ci/evaluation_soft_mode_smoke_summary.json"
    )
