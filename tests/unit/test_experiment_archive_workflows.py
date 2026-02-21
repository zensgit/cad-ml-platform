"""Safety regression tests for experiment archive GitHub workflows."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DRY_RUN_WORKFLOW = ROOT / ".github" / "workflows" / "experiment-archive-dry-run.yml"
APPLY_WORKFLOW = ROOT / ".github" / "workflows" / "experiment-archive-apply.yml"


def _load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_archive_dry_run_workflow_has_expected_triggers_and_guards() -> None:
    workflow = _load_workflow(DRY_RUN_WORKFLOW)

    assert workflow["name"] == "Experiment Archive Dry Run"
    assert workflow["on"]["schedule"][0]["cron"] == "30 2 * * *"

    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["experiments_root"]["default"] == "reports/experiments"
    assert dispatch_inputs["keep_latest_days"]["default"] == "7"

    run_step = _get_step(workflow, "dry-run", "Run archive dry-run")
    run_script = run_step["run"]
    assert "scripts/ci/archive_experiment_dirs.py" in run_script
    assert "--dry-run" in run_script
    assert "--manifest-json" in run_script

    _get_step(workflow, "dry-run", "Upload dry-run manifest")
    _get_step(workflow, "dry-run", "Upload dry-run log")


def test_archive_apply_workflow_has_manual_approval_and_delete_guards() -> None:
    workflow = _load_workflow(APPLY_WORKFLOW)

    assert workflow["name"] == "Experiment Archive Apply"
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["approval_phrase"]["required"] == "true"

    validate_step = _get_step(workflow, "validate-approval", "Validate approval phrase")
    validate_script = validate_step["run"]
    assert "I_UNDERSTAND_DELETE_SOURCE" in validate_script
    assert "exit 1" in validate_script

    apply_job = workflow["jobs"]["apply"]
    assert apply_job["needs"] == "validate-approval"
    assert apply_job["environment"] == "experiment-archive-approval"

    run_step = _get_step(workflow, "apply", "Apply archive with delete-source")
    run_script = run_step["run"]
    assert "scripts/ci/archive_experiment_dirs.py" in run_script
    assert "--delete-source" in run_script
    assert "ARGS+=(--require-exists)" in run_script

    _get_step(workflow, "apply", "Upload apply manifest")
    _get_step(workflow, "apply", "Upload apply log")
    _get_step(workflow, "apply", "Upload generated archives")
