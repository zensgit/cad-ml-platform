"""Regression checks for hybrid superpass gate wiring in evaluation-report workflow."""

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


def _assert_empty_workflow_dispatch(workflow: dict) -> None:
    assert workflow["on"]["workflow_dispatch"] == {}


def _load_bash_helper_from_step(step: dict) -> str:
    run = step["run"]
    match = re.search(r"bash\s+(scripts/ci/[^\s]+)", run)
    if not match:
        return run
    return (ROOT / match.group(1)).read_text(encoding="utf-8")


def test_workflow_has_hybrid_superpass_inputs_and_env() -> None:
    workflow = _load_workflow()
    _assert_empty_workflow_dispatch(workflow)
    env = workflow["env"]

    assert "HYBRID_BLIND_ENABLE" in env
    assert "HYBRID_BLIND_GATE_CONFIG" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_ENABLE" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_OUTPUT_JSON" in env
    assert "HYBRID_SUPERPASS_ENABLE" in env
    assert "HYBRID_SUPERPASS_CONFIG" in env
    assert "HYBRID_SUPERPASS_OUTPUT_JSON" in env
    assert "HYBRID_SUPERPASS_MISSING_MODE" in env
    assert "HYBRID_SUPERPASS_VALIDATION_JSON" in env
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in env


def test_workflow_has_hybrid_superpass_steps_and_artifacts() -> None:
    workflow = _load_workflow()

    blind_step = _get_step(workflow, "evaluate", "Run Hybrid blind benchmark (optional)")
    assert "scripts/ci/build_hybrid_blind_synthetic_dxf_dataset.py" in blind_step["run"]
    blind_gate_step = _get_step(workflow, "evaluate", "Check Hybrid blind gate (optional)")
    blind_gate_script = blind_gate_step["run"]
    assert "scripts/ci/check_hybrid_blind_gate.py" in blind_gate_script
    assert "--dataset-source" in blind_gate_script
    assert "steps.hybrid_blind_eval.outputs.dataset_source" in blind_gate_script
    calibration_step = _get_step(workflow, "evaluate", "Calibrate Hybrid confidence from review CSV (optional)")
    assert "scripts/calibrate_hybrid_confidence.py" in calibration_step["run"]

    gate_step = _get_step(workflow, "evaluate", "Check Hybrid superpass gate (optional)")
    gate_script = gate_step["run"]
    assert "scripts/ci/check_hybrid_superpass_targets.py" in gate_script
    assert "steps.hybrid_blind_gate.outputs.report_path" in gate_script
    assert "steps.hybrid_blind_eval.outputs.dataset_source" in gate_script
    assert "steps.hybrid_calibration.outputs.output_json" in gate_script
    assert "--hybrid-blind-gate-report" in gate_script
    assert "--hybrid-blind-dataset-source" in gate_script
    assert "--hybrid-calibration-json" in gate_script
    assert "--config" in gate_script
    assert "--missing-mode" in gate_script
    assert "--output" in gate_script

    strict_step = _get_step(
        workflow, "evaluate", "Evaluate Hybrid superpass strict mode (optional)"
    )
    strict_script = strict_step["run"]
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in strict_script
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

    upload_step = _get_step(workflow, "evaluate", "Upload Hybrid superpass gate artifact")
    assert upload_step["if"] == "steps.hybrid_superpass_gate.outputs.enabled == 'true'"
    assert "report_path" in upload_step["with"]["path"]

    summary_step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = _load_bash_helper_from_step(summary_step)
    assert "Hybrid superpass gate status" in summary_script
    assert "Hybrid superpass gate strict_should_fail" in summary_script
