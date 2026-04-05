"""Regression checks for release summary wiring in evaluation-report workflow."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "evaluation-report.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def _step_names(workflow: dict, job_name: str) -> list[str]:
    return [s.get("name", "") for s in workflow["jobs"][job_name]["steps"]]


def test_generate_release_summary_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Generate eval reporting release summary")

    assert step.get("if") == "always()"
    assert "generate_eval_reporting_release_summary.py" in step["run"]
    assert "--index-json" in step["run"]
    assert "--stack-summary-json" in step["run"]


def test_append_release_summary_to_job_summary_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Append eval reporting release summary to job summary")

    assert step.get("if") == "always()"
    assert "GITHUB_STEP_SUMMARY" in step["run"]
    assert "eval_reporting_release_summary.md" in step["run"]


def test_upload_release_summary_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload eval reporting release summary")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_release_summary" in upload_path


def test_release_summary_steps_after_stack_summary_before_fail() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    stack_upload_idx = names.index("Upload eval reporting stack summary")
    gen_idx = names.index("Generate eval reporting release summary")
    append_idx = names.index("Append eval reporting release summary to job summary")
    upload_idx = names.index("Upload eval reporting release summary")
    fail_idx = names.index("Fail workflow on refresh failure")

    assert gen_idx > stack_upload_idx
    assert append_idx > gen_idx
    assert upload_idx > append_idx
    assert fail_idx > upload_idx


# --- Batch 9B: status check step ---


def test_status_check_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Post Eval Reporting status check")

    assert step.get("if") == "always()"
    assert step.get("continue-on-error") in (True, "true")
    script = str(step.get("with", {}).get("script", ""))
    assert "post_eval_reporting_status_check.js" in script
    assert "postEvalReportingStatusCheck" in script


def test_status_check_step_after_release_summary_before_fail() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    upload_idx = names.index("Upload eval reporting release summary")
    check_idx = names.index("Post Eval Reporting status check")
    fail_idx = names.index("Fail workflow on refresh failure")

    assert check_idx > upload_idx
    assert fail_idx > check_idx


def test_status_check_step_consumes_release_summary_path() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Post Eval Reporting status check")

    script = str(step.get("with", {}).get("script", ""))
    assert "eval_reporting_release_summary.json" in script
