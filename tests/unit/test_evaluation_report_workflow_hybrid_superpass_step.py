"""Regression checks for hybrid superpass gate wiring in evaluation-report workflow."""

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


def test_workflow_has_hybrid_superpass_inputs_and_env() -> None:
    workflow = _load_workflow()
    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    env = workflow["env"]

    assert "hybrid_superpass_enable" in dispatch_inputs
    assert "hybrid_superpass_missing_mode" in dispatch_inputs
    assert "hybrid_superpass_fail_on_failed" in dispatch_inputs

    assert "EVALUATION_STRICT_FAIL_MODE" in env
    assert "HYBRID_SUPERPASS_ENABLE" in env
    assert "HYBRID_SUPERPASS_CONFIG" in env
    assert "HYBRID_SUPERPASS_OUTPUT_JSON" in env
    assert "HYBRID_SUPERPASS_MISSING_MODE" in env
    assert "HYBRID_SUPERPASS_VALIDATION_JSON" in env
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in env
    assert "HYBRID_SUPERPASS_VALIDATION_STRICT" in env
    assert "HYBRID_SUPERPASS_VALIDATION_SCHEMA_MODE" in env


def test_workflow_has_hybrid_superpass_steps_and_artifacts() -> None:
    workflow = _load_workflow()

    resolve_mode_step = _get_step(workflow, "evaluate", "Resolve strict gate fail mode")
    assert resolve_mode_step["id"] == "strict_fail_mode"
    assert "MODE_VALUE" in resolve_mode_step["run"]
    assert 'token" == "soft"' in resolve_mode_step["run"]
    assert 'echo "mode=$mode"' in resolve_mode_step["run"]

    gate_step = _get_step(
        workflow, "evaluate", "Check Hybrid superpass gate (optional)"
    )
    gate_script = gate_step["run"]
    assert "scripts/ci/check_hybrid_superpass_targets.py" in gate_script
    assert "--hybrid-blind-gate-report" in gate_script
    assert "--hybrid-calibration-json" in gate_script
    assert "--config" in gate_script
    assert "--missing-mode" in gate_script
    assert "--output" in gate_script

    strict_step = _get_step(
        workflow, "evaluate", "Evaluate Hybrid superpass strict mode (optional)"
    )
    strict_script = strict_step["run"]
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in strict_script
    assert "hybrid_superpass_fail_on_failed" in strict_script
    assert "status is not passed" in strict_script

    final_fail = _get_step(
        workflow,
        "evaluate",
        "Fail workflow when Hybrid superpass strict check requires blocking",
    )
    assert (
        final_fail["if"]
        == "steps.hybrid_superpass_gate_strict.outputs.should_fail == 'true' && steps.strict_fail_mode.outputs.mode != 'soft'"
    )

    validate_step = _get_step(
        workflow, "evaluate", "Validate Hybrid superpass report structure (optional)"
    )
    assert (
        validate_step["if"] == "steps.hybrid_superpass_gate.outputs.enabled == 'true'"
    )
    validate_script = validate_step["run"]
    assert "scripts/ci/validate_hybrid_superpass_reports.py" in validate_script
    assert "--superpass-json" in validate_script
    assert "--hybrid-blind-gate-report" in validate_script
    assert "--hybrid-calibration-json" in validate_script
    assert "--output-json" in validate_script
    assert "--schema-mode" in validate_script

    upload_step = _get_step(
        workflow, "evaluate", "Upload Hybrid superpass gate artifact"
    )
    assert upload_step["if"] == "steps.hybrid_superpass_gate.outputs.enabled == 'true'"
    assert "report_path" in upload_step["with"]["path"]
    assert (
        "steps.hybrid_superpass_validate.outputs.report_path"
        in upload_step["with"]["path"]
    )

    summary_step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = summary_step["run"]
    assert "Hybrid superpass gate status" in summary_script
    assert "Hybrid superpass gate strict_should_fail" in summary_script
    assert "Hybrid superpass structure validation status" in summary_script
    assert "Hybrid superpass structure validation strict_mode" in summary_script
    assert "Hybrid superpass structure validation schema_mode" in summary_script
