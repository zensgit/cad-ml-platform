from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hybrid-superpass-nightly.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    for step in workflow["jobs"][job_name]["steps"]:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"missing step={step_name!r} in job={job_name!r}")


def test_workflow_has_daily_schedule_and_manual_dispatch_inputs() -> None:
    workflow = _load_workflow()

    assert workflow["name"] == "Hybrid Superpass Nightly"
    assert "dispatch_trace_id" in workflow["run-name"]
    assert workflow["on"]["schedule"][0]["cron"] == "15 2 * * *"

    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["target_repo"]["default"] == ""
    assert inputs["target_ref"]["default"] == "main"
    assert inputs["target_workflow"]["default"] == "hybrid-superpass-e2e.yml"
    assert inputs["dispatch_trace_id"]["default"] == ""
    assert inputs["dual_wait_timeout_seconds"]["default"] == "900"
    assert inputs["dual_poll_interval_seconds"]["default"] == "3"
    assert inputs["dual_list_limit"]["default"] == "20"
    assert inputs["strict_require_distinct_run_ids"]["default"] == "true"
    assert inputs["strict_require_trace_pair"]["default"] == "true"


def test_workflow_permissions_and_default_repo_ref_wiring() -> None:
    workflow = _load_workflow()

    permissions = workflow["permissions"]
    assert permissions["actions"] == "write"
    assert permissions["contents"] == "read"
    assert workflow["env"]["FORCE_JAVASCRIPT_ACTIONS_TO_NODE24"] == "true"

    env = workflow["jobs"]["nightly-superpass"]["env"]
    assert "github.event.inputs.target_repo" in env["TARGET_REPO"]
    assert "github.repository" in env["TARGET_REPO"]
    assert "github.event.inputs.target_ref" in env["TARGET_REF"]
    assert "'main'" in env["TARGET_REF"]
    assert "github.event.inputs.dispatch_trace_id" in env["NIGHTLY_TRACE_ID"]
    assert "nsp-" in env["NIGHTLY_TRACE_ID"]
    assert "github.run_id" in env["NIGHTLY_TRACE_ID"]
    assert "github.event.inputs.dual_wait_timeout_seconds" in env["DUAL_WAIT_TIMEOUT_SECONDS"]
    assert "'900'" in env["DUAL_WAIT_TIMEOUT_SECONDS"]
    assert "github.event.inputs.dual_poll_interval_seconds" in env["DUAL_POLL_INTERVAL_SECONDS"]
    assert "'3'" in env["DUAL_POLL_INTERVAL_SECONDS"]
    assert "github.event.inputs.dual_list_limit" in env["DUAL_LIST_LIMIT"]
    assert "'20'" in env["DUAL_LIST_LIMIT"]
    assert (
        "github.event.inputs.strict_require_distinct_run_ids"
        in env["STRICT_REQUIRE_DISTINCT_RUN_IDS"]
    )
    assert "'true'" in env["STRICT_REQUIRE_DISTINCT_RUN_IDS"]
    assert "github.event.inputs.strict_require_trace_pair" in env["STRICT_REQUIRE_TRACE_PAIR"]
    assert "'true'" in env["STRICT_REQUIRE_TRACE_PAIR"]


def test_workflow_has_dual_dispatch_compare_artifact_and_summary_steps() -> None:
    workflow = _load_workflow()
    job_name = "nightly-superpass"

    dual_step = _get_step(workflow, job_name, "Run nightly dual dispatch and compare")
    dual_script = dual_step["run"]
    assert dual_step["if"] == "always()"
    assert "scripts/ci/run_hybrid_superpass_dual_dispatch.py" in dual_script
    assert "--workflow \"$TARGET_WORKFLOW\"" in dual_script
    assert "--repo \"$TARGET_REPO\"" in dual_script
    assert "--ref \"$TARGET_REF\"" in dual_script
    assert "--wait-timeout-seconds \"$DUAL_WAIT_TIMEOUT_SECONDS\"" in dual_script
    assert "--poll-interval-seconds \"$DUAL_POLL_INTERVAL_SECONDS\"" in dual_script
    assert "--list-limit \"$DUAL_LIST_LIMIT\"" in dual_script
    assert "--fail-output-json \"$FAIL_JSON\"" in dual_script
    assert "--success-output-json \"$SUCCESS_JSON\"" in dual_script
    assert "--compare-output-json \"$COMPARE_JSON\"" in dual_script
    assert "--compare-output-md \"$COMPARE_MD\"" in dual_script
    assert "--output-json \"$DUAL_SUMMARY_JSON\"" in dual_script
    assert "--dispatch-trace-prefix \"$NIGHTLY_TRACE_ID\"" in dual_script
    assert "strict_flags=()" in dual_script
    assert "STRICT_REQUIRE_DISTINCT_RUN_IDS,,}" in dual_script
    assert "strict_flags+=(--strict-require-distinct-run-ids)" in dual_script
    assert "STRICT_REQUIRE_TRACE_PAIR,,}" in dual_script
    assert "strict_flags+=(--strict-require-trace-pair)" in dual_script
    assert "--strict" in dual_script
    assert "\"${strict_flags[@]}\"" in dual_script

    render_step = _get_step(workflow, job_name, "Render nightly dual summary markdown")
    render_script = render_step["run"]
    assert render_step["if"] == "always()"
    assert "scripts/ci/render_hybrid_superpass_dual_summary.py" in render_script
    assert "--dual-summary-json \"$DUAL_SUMMARY_JSON\"" in render_script
    assert "compare_json_args=()" in render_script
    assert "compare_json_args+=(--compare-json \"$COMPARE_JSON\")" in render_script
    assert "compare_md_args=()" in render_script
    assert "compare_md_args+=(--compare-md \"$COMPARE_MD\")" in render_script
    assert "\"${compare_json_args[@]}\"" in render_script
    assert "\"${compare_md_args[@]}\"" in render_script
    assert "--output-md \"$DUAL_SUMMARY_MD\"" in render_script
    assert "dual_summary_md=$DUAL_SUMMARY_MD" in render_script

    upload_step = _get_step(
        workflow, job_name, "Upload nightly superpass compare artifacts"
    )
    assert upload_step["if"] == "always()"
    assert upload_step["uses"] == "actions/upload-artifact@v4"
    assert "${{ env.FAIL_JSON }}" in upload_step["with"]["path"]
    assert "${{ env.SUCCESS_JSON }}" in upload_step["with"]["path"]
    assert "${{ env.COMPARE_JSON }}" in upload_step["with"]["path"]
    assert "${{ env.COMPARE_MD }}" in upload_step["with"]["path"]
    assert "${{ env.DUAL_SUMMARY_JSON }}" in upload_step["with"]["path"]
    assert "${{ env.DUAL_SUMMARY_MD }}" in upload_step["with"]["path"]

    summary_step = _get_step(workflow, job_name, "Write nightly superpass step summary")
    summary_script = summary_step["run"]
    assert summary_step["if"] == "always()"
    assert "GITHUB_STEP_SUMMARY" in summary_script
    assert "Trace ID" in summary_script
    assert "$NIGHTLY_TRACE_ID" in summary_script
    assert "Dual wait timeout seconds" in summary_script
    assert "$DUAL_WAIT_TIMEOUT_SECONDS" in summary_script
    assert "Dual poll interval seconds" in summary_script
    assert "$DUAL_POLL_INTERVAL_SECONDS" in summary_script
    assert "Dual list limit" in summary_script
    assert "$DUAL_LIST_LIMIT" in summary_script
    assert "Strict require distinct run ids" in summary_script
    assert "$STRICT_REQUIRE_DISTINCT_RUN_IDS" in summary_script
    assert "Strict require trace pair" in summary_script
    assert "$STRICT_REQUIRE_TRACE_PAIR" in summary_script
    assert "Dual dispatch step outcome" in summary_script
    assert "Dual summary JSON" in summary_script
    assert "Dual summary Markdown" in summary_script
    assert "$DUAL_SUMMARY_MD" in summary_script
