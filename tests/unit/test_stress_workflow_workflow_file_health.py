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
    assert "scripts/ci/check_workflow_publish_helper_adoption.py" in push_paths
    assert "scripts/ci/generate_workflow_inventory_report.py" in push_paths
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

    inventory_step = _get_step(
        workflow, "workflow-file-health", "Generate workflow inventory audit report"
    )
    inventory_script = inventory_step["run"]
    assert "scripts/ci/generate_workflow_inventory_report.py" in inventory_script
    assert '--workflow-root ".github/workflows"' in inventory_script
    assert "--ci-watch-required-workflows" in inventory_script
    assert "--output-json reports/ci/workflow_inventory_report.json" in inventory_script
    assert "--output-md reports/ci/workflow_inventory_report.md" in inventory_script
    assert ">/dev/null" in inventory_script

    publish_helper_step = _get_step(
        workflow,
        "workflow-file-health",
        "Generate workflow publish helper adoption report",
    )
    publish_helper_script = publish_helper_step["run"]
    assert "scripts/ci/check_workflow_publish_helper_adoption.py" in publish_helper_script
    assert '--workflow-root ".github/workflows"' in publish_helper_script
    assert "--summary-json-out reports/ci/workflow_publish_helper_adoption.json" in publish_helper_script
    assert "--output-md reports/ci/workflow_publish_helper_adoption.md" in publish_helper_script
    assert ">/dev/null" in publish_helper_script

    upload = _get_step(workflow, "workflow-file-health", "Upload workflow health summary")
    assert (
        upload["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert upload["with"]["name"] == "workflow-file-health-${{ github.run_number }}"
    assert upload["with"]["path"] == "reports/ci/workflow_file_health_summary.json"

    inventory_upload = _get_step(
        workflow, "workflow-file-health", "Upload workflow inventory audit report"
    )
    assert (
        inventory_upload["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert (
        inventory_upload["with"]["name"]
        == "workflow-inventory-report-${{ github.run_number }}"
    )
    inventory_upload_path = inventory_upload["with"]["path"]
    assert "reports/ci/workflow_inventory_report.json" in inventory_upload_path
    assert "reports/ci/workflow_inventory_report.md" in inventory_upload_path

    publish_helper_upload = _get_step(
        workflow,
        "workflow-file-health",
        "Upload workflow publish helper adoption report",
    )
    assert (
        publish_helper_upload["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert (
        publish_helper_upload["with"]["name"]
        == "workflow-publish-helper-adoption-${{ github.run_number }}"
    )
    publish_helper_upload_path = publish_helper_upload["with"]["path"]
    assert "reports/ci/workflow_publish_helper_adoption.json" in publish_helper_upload_path
    assert "reports/ci/workflow_publish_helper_adoption.md" in publish_helper_upload_path

    append_step = _get_step(
        workflow, "workflow-file-health", "Append workflow inventory summary"
    )
    append_script = append_step["run"]
    assert "cat reports/ci/workflow_inventory_report.md" in append_script
    assert "workflow inventory markdown missing" in append_script
    assert '>> "$GITHUB_STEP_SUMMARY"' in append_script

    append_publish_helper_step = _get_step(
        workflow,
        "workflow-file-health",
        "Append workflow publish helper adoption summary",
    )
    append_publish_helper_script = append_publish_helper_step["run"]
    assert "cat reports/ci/workflow_publish_helper_adoption.md" in append_publish_helper_script
    assert "workflow publish helper markdown missing" in append_publish_helper_script
    assert '>> "$GITHUB_STEP_SUMMARY"' in append_publish_helper_script


def test_stress_jobs_depend_on_workflow_file_health() -> None:
    workflow = _load_workflow()
    assert workflow["jobs"]["metrics-consistency"]["needs"] == "workflow-file-health"
    assert workflow["jobs"]["stress-unit-tests"]["needs"] == "workflow-file-health"
