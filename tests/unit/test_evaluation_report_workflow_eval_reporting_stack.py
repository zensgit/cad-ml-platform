"""Regression checks for eval reporting stack wiring in evaluation-report workflow."""

from pathlib import Path
import re

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


def _load_bash_helper_from_step(step: dict) -> str:
    run = step["run"]
    match = re.search(r"bash\s+(scripts/ci/[^\s]+)", run)
    if not match:
        return run
    return (ROOT / match.group(1)).read_text(encoding="utf-8")


def _step_names(workflow: dict, job_name: str) -> list[str]:
    return [s.get("name", "") for s in workflow["jobs"][job_name]["steps"]]


# --- env vars ---


def test_workflow_env_includes_eval_reporting_stack_vars() -> None:
    workflow = _load_workflow()
    env = workflow["env"]

    assert env["REPORT_PATH"] == "reports/eval_history/report_static"
    assert env["INTERACTIVE_REPORT_PATH"] == "reports/eval_history/report_interactive"
    assert env["EVAL_REPORTING_BUNDLE_JSON"] == "reports/eval_history/eval_reporting_bundle.json"
    assert env["EVAL_REPORTING_BUNDLE_HEALTH_JSON"] == "reports/eval_history/eval_reporting_bundle_health_report.json"
    assert env["EVAL_REPORTING_INDEX_JSON"] == "reports/eval_history/eval_reporting_index.json"
    assert "EVAL_REPORTING_REFRESH_DAYS" in env


def test_static_and_interactive_report_paths_are_distinct() -> None:
    workflow = _load_workflow()
    env = workflow["env"]

    assert env["REPORT_PATH"] != env["INTERACTIVE_REPORT_PATH"]
    assert "report_static" in env["REPORT_PATH"]
    assert "report_interactive" in env["INTERACTIVE_REPORT_PATH"]


# --- refresh step ---


def test_refresh_step_exists_and_uses_continue_on_error() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Refresh eval reporting stack")

    assert step.get("continue-on-error") in (True, "true"), (
        "refresh step must use continue-on-error to preserve diagnostics"
    )
    assert "refresh_eval_reporting_stack.py" in step["run"]
    assert step.get("id") == "eval_reporting_refresh"


def test_refresh_step_captures_exit_code() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Refresh eval reporting stack")

    assert "exit_code" in step["run"], "refresh step must capture exit_code"
    assert "GITHUB_OUTPUT" in step["run"]


# --- old steps removed ---


def test_old_generate_steps_removed() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    assert "Generate HTML report" not in names
    assert "Generate weekly rolling summary" not in names


# --- artifact uploads ---


def test_evaluation_report_artifact_upload_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload evaluation report")

    assert "REPORT_PATH" in str(step.get("with", {}).get("path", ""))


def test_interactive_report_artifact_upload_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload interactive report")

    assert "INTERACTIVE_REPORT_PATH" in str(step.get("with", {}).get("path", ""))


def test_eval_reporting_stack_artifact_upload_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload eval reporting stack artifacts")

    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_bundle.json" in upload_path or "EVAL_REPORTING_BUNDLE_JSON" in upload_path
    assert "eval_reporting_bundle_health_report" in upload_path or "EVAL_REPORTING_BUNDLE_HEALTH_JSON" in upload_path
    assert "eval_reporting_index" in upload_path or "EVAL_REPORTING_INDEX_JSON" in upload_path


# --- fail step after uploads ---


def test_fail_step_exists_after_uploads() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    fail_idx = names.index("Fail workflow on refresh failure")
    eval_upload_idx = names.index("Upload evaluation report")
    interactive_upload_idx = names.index("Upload interactive report")
    stack_upload_idx = names.index("Upload eval reporting stack artifacts")

    assert fail_idx > eval_upload_idx
    assert fail_idx > interactive_upload_idx
    assert fail_idx > stack_upload_idx


def test_fail_step_checks_refresh_exit_code() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Fail workflow on refresh failure")

    assert "eval_reporting_refresh" in str(step.get("if", ""))
    assert "exit_code" in str(step.get("if", ""))
    assert "exit 1" in step["run"]


# --- Batch 5B: summary / annotation / STEP_SUMMARY ---


def test_stack_summary_step_exists_and_always_runs() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Summarize eval reporting stack status")

    assert step.get("if") == "always()"
    assert "summarize_eval_reporting_stack_status.py" in step["run"]
    assert step.get("id") == "eval_reporting_stack_summary"


def test_append_to_job_summary_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Append eval reporting stack summary to job summary")

    assert step.get("if") == "always()"
    assert "GITHUB_STEP_SUMMARY" in step["run"]
    assert "eval_reporting_stack_summary.md" in step["run"]


def test_stack_summary_upload_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload eval reporting stack summary")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_stack_summary" in upload_path


def test_summary_and_upload_before_fail_step() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    fail_idx = names.index("Fail workflow on refresh failure")
    summary_idx = names.index("Summarize eval reporting stack status")
    append_idx = names.index("Append eval reporting stack summary to job summary")
    upload_idx = names.index("Upload eval reporting stack summary")

    assert summary_idx < fail_idx
    assert append_idx < fail_idx
    assert upload_idx < fail_idx


def test_landing_page_upload_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload landing page")

    upload_path = str(step.get("with", {}).get("path", ""))
    assert "index.html" in upload_path


def test_landing_page_in_stack_artifacts() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload eval reporting stack artifacts")

    upload_path = str(step.get("with", {}).get("path", ""))
    assert "index.html" in upload_path


def test_pr_comment_step_has_stack_summary_env() -> None:
    workflow = _load_workflow()
    comment_step = _get_step(workflow, "evaluate", "Comment PR with results")
    step_env = comment_step.get("env", {})
    assert "EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT" in step_env
    assert "EVAL_REPORTING_INDEX_JSON_FOR_COMMENT" in step_env


def test_notify_step_passes_stack_summary_and_index() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")
    # Find the step containing notify_eval_results
    steps = workflow["jobs"]["evaluate"]["steps"]
    notify_step = None
    for step in steps:
        run_text = str(step.get("run", ""))
        if "notify_eval_results.py" in run_text:
            notify_step = step
            break
    assert notify_step is not None, "notify step not found"
    run_text = notify_step["run"]
    assert "--stack-summary-json" in run_text
    assert "--index-json" in run_text
    assert "eval_reporting_stack_summary.json" in run_text
    assert "eval_reporting_index.json" in run_text


def test_stale_weekly_summary_reference_removed() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = _load_bash_helper_from_step(step)

    assert "weekly_summary.outputs.output_md" not in step["run"]
    assert "eval_reporting_stack_summary" in summary_script or "eval_reporting_stack" in summary_script
