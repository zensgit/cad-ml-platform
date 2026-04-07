from __future__ import annotations

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "evaluation-report.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    for step in workflow["jobs"][job_name]["steps"]:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"missing step={step_name!r} in job={job_name!r}")


def _assert_empty_workflow_dispatch(workflow: dict) -> None:
    assert workflow["on"]["workflow_dispatch"] == {}


def _load_bash_helper_from_step(step: dict) -> str:
    run = step["run"]
    match = re.search(r"bash\s+(scripts/ci/[^\s]+)", run)
    if not match:
        return run
    return (ROOT / match.group(1)).read_text(encoding="utf-8")


def test_workflow_dispatch_and_env_expose_hybrid_superpass_controls() -> None:
    workflow = _load_workflow()
    _assert_empty_workflow_dispatch(workflow)

    env = workflow["env"]
    assert "HYBRID_BLIND_ENABLE" in env
    assert "HYBRID_BLIND_DXF_DIR" in env
    assert "HYBRID_BLIND_GATE_CONFIG" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_ENABLE" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_INPUT_CSV" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_OUTPUT_JSON" in env
    assert "HYBRID_SUPERPASS_ENABLE" in env
    assert "HYBRID_SUPERPASS_CONFIG" in env
    assert "HYBRID_SUPERPASS_OUTPUT_JSON" in env
    assert "HYBRID_SUPERPASS_MISSING_MODE" in env
    assert "HYBRID_SUPERPASS_GATE_REPORT_JSON" in env
    assert "HYBRID_SUPERPASS_CALIBRATION_JSON" in env
    assert "HYBRID_SUPERPASS_VALIDATION_JSON" in env
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in env


def test_workflow_materializes_hybrid_superpass_prerequisites() -> None:
    workflow = _load_workflow()

    blind_step = _get_step(workflow, "evaluate", "Run Hybrid blind benchmark (optional)")
    blind_script = blind_step["run"]
    assert "scripts/ci/build_hybrid_blind_synthetic_dxf_dataset.py" in blind_script
    assert "scripts/batch_analyze_dxf_local.py" in blind_script
    assert "HYBRID_BLIND_ENABLE" in blind_script

    blind_gate_step = _get_step(workflow, "evaluate", "Check Hybrid blind gate (optional)")
    blind_gate_script = blind_gate_step["run"]
    assert "scripts/ci/check_hybrid_blind_gate.py" in blind_gate_script
    assert "--summary-json" in blind_gate_script
    assert "--config" in blind_gate_script
    assert "--output" in blind_gate_script

    calibration_step = _get_step(workflow, "evaluate", "Calibrate Hybrid confidence from review CSV (optional)")
    calibration_script = calibration_step["run"]
    assert "scripts/calibrate_hybrid_confidence.py" in calibration_script
    assert "HYBRID_CONFIDENCE_CALIBRATION_ENABLE" in calibration_script
    assert "steps.graph2d_review_pack.outputs.input_csv" in calibration_script


def test_workflow_has_optional_hybrid_superpass_gate_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Check Hybrid superpass gate (optional)")
    assert step["if"] == "always()"
    run_script = step["run"]
    assert "scripts/ci/check_hybrid_superpass_targets.py" in run_script
    assert "steps.hybrid_blind_gate.outputs.report_path" in run_script
    assert "steps.hybrid_calibration.outputs.output_json" in run_script
    assert "--hybrid-blind-gate-report" in run_script
    assert "--hybrid-calibration-json" in run_script
    assert "--config" in run_script
    assert "--missing-mode" in run_script
    assert "--output" in run_script
    assert "HYBRID_SUPERPASS_ENABLE" in run_script
    assert "HYBRID_SUPERPASS_MISSING_MODE" in run_script
    assert "status=" in run_script
    assert "headline=" in run_script


def test_workflow_uploads_superpass_artifact_and_summary_lines() -> None:
    workflow = _load_workflow()
    upload_step = _get_step(workflow, "evaluate", "Upload Hybrid superpass gate artifact")
    assert upload_step["if"] == "steps.hybrid_superpass_gate.outputs.enabled == 'true'"
    assert (
        upload_step["with"]["name"]
        == "hybrid-superpass-gate-${{ github.run_number }}"
    )
    assert "steps.hybrid_superpass_gate.outputs.report_path" in upload_step["with"]["path"]

    summary_step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = _load_bash_helper_from_step(summary_step)
    assert "Hybrid superpass gate status" in summary_script
    assert "Hybrid superpass gate headline" in summary_script
    assert "Hybrid superpass gate report" in summary_script
    assert "Hybrid superpass gate missing_mode" in summary_script


def test_workflow_has_superpass_strict_mode_steps() -> None:
    workflow = _load_workflow()
    strict_step = _get_step(workflow, "evaluate", "Evaluate Hybrid superpass strict mode (optional)")
    strict_script = strict_step["run"]
    assert "HYBRID_SUPERPASS_FAIL_ON_FAILED" in strict_script
    assert "status is not passed" in strict_script

    final_fail_step = _get_step(
        workflow,
        "evaluate",
        "Fail workflow when Hybrid superpass strict check requires blocking",
    )
    assert (
        final_fail_step["if"]
        == "steps.hybrid_superpass_gate_strict.outputs.should_fail == 'true' && steps.strict_fail_mode.outputs.mode != 'soft'"
    )
    assert "Failure reason" in final_fail_step["run"]
