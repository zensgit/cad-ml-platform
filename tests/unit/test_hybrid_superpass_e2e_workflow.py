from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hybrid-superpass-e2e.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_hybrid_superpass_e2e_workflow_dispatch_inputs() -> None:
    workflow = _load_workflow()
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    run_name = workflow.get("run-name", "")
    env = workflow.get("env", {})

    assert "hybrid_superpass_enable" in dispatch_inputs
    assert "hybrid_superpass_missing_mode" in dispatch_inputs
    assert "hybrid_superpass_fail_on_failed" in dispatch_inputs
    assert "hybrid_blind_gate_report_json" in dispatch_inputs
    assert "hybrid_calibration_json" in dispatch_inputs
    assert "hybrid_superpass_config" in dispatch_inputs
    assert "hybrid_superpass_output_json" in dispatch_inputs
    assert "dispatch_trace_id" in dispatch_inputs
    assert "dispatch_trace_id" in str(run_name)
    assert env.get("FORCE_JAVASCRIPT_ACTIONS_TO_NODE24") == "true"


def test_hybrid_superpass_e2e_workflow_contains_gate_and_strict_steps() -> None:
    workflow = _load_workflow()

    gate = _get_step(workflow, "superpass", "Run Hybrid superpass gate")
    gate_script = gate["run"]
    assert "scripts/ci/check_hybrid_superpass_targets.py" in gate_script
    assert "--hybrid-blind-gate-report" in gate_script
    assert "--hybrid-calibration-json" in gate_script
    assert "--config" in gate_script
    assert "--missing-mode" in gate_script
    assert "--output" in gate_script

    strict = _get_step(workflow, "superpass", "Evaluate Hybrid superpass strict mode")
    strict_script = strict["run"]
    assert "hybrid_superpass_fail_on_failed" in strict_script
    assert "status" in strict_script

    fail_step = _get_step(
        workflow,
        "superpass",
        "Fail workflow when Hybrid superpass strict check requires blocking",
    )
    assert fail_step["if"] == "steps.hybrid_superpass_gate_strict.outputs.should_fail == 'true'"


def test_hybrid_superpass_e2e_workflow_upload_and_summary() -> None:
    workflow = _load_workflow()

    upload = _get_step(workflow, "superpass", "Upload Hybrid superpass gate artifact")
    assert upload["if"] == "steps.hybrid_superpass_gate.outputs.enabled == 'true'"
    assert "steps.hybrid_superpass_gate.outputs.report_path" in upload["with"]["path"]

    summary = _get_step(workflow, "superpass", "Create job summary")
    summary_script = summary["run"]
    assert "Hybrid Superpass E2E" in summary_script
    assert "Gate status" in summary_script
    assert "Strict should fail" in summary_script
