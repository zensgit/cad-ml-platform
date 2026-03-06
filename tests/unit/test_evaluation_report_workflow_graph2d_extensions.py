"""Regression checks for Graph2D extensions in evaluation-report workflow."""

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


def test_workflow_env_includes_graph2d_review_and_train_sweep_flags() -> None:
    workflow = _load_workflow()
    env = workflow["env"]

    assert "GRAPH2D_REVIEW_PACK_INPUT_CSV" in env
    assert "GRAPH2D_REVIEW_PACK_OUTPUT_CSV" in env
    assert "GRAPH2D_REVIEW_PACK_SUMMARY_JSON" in env
    assert "GRAPH2D_REVIEW_PACK_LOW_CONF_THRESHOLD" in env
    assert "GRAPH2D_REVIEW_PACK_TOP_K" in env
    assert "GRAPH2D_REVIEW_PACK_GATE_CONFIG" in env
    assert "GRAPH2D_REVIEW_PACK_GATE_REPORT" in env
    assert "GRAPH2D_TRAIN_SWEEP_ENABLE" in env
    assert "GRAPH2D_TRAIN_SWEEP_EXECUTE" in env
    assert "GRAPH2D_TRAIN_SWEEP_FAIL_ON_ERROR" in env
    assert "GRAPH2D_TRAIN_SWEEP_RECIPES" in env
    assert "GRAPH2D_TRAIN_SWEEP_SEEDS" in env
    assert "GRAPH2D_TRAIN_SWEEP_BASE_ARGS_JSON" in env

    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "review_gate_min_total_rows" in dispatch_inputs
    assert "review_gate_max_candidate_rate" in dispatch_inputs
    assert "review_gate_max_hybrid_rejected_rate" in dispatch_inputs
    assert "review_gate_max_conflict_rate" in dispatch_inputs
    assert "review_gate_max_low_confidence_rate" in dispatch_inputs
    assert "review_gate_strict" in dispatch_inputs
    assert "review_pack_input_csv" in dispatch_inputs


def test_workflow_has_optional_graph2d_review_pack_and_train_sweep_steps() -> None:
    workflow = _load_workflow()
    review_step = _get_step(workflow, "evaluate", "Build hybrid rejection review pack (optional)")
    review_script = review_step["run"]
    assert "scripts/export_hybrid_rejection_review_pack.py" in review_script
    assert "github.event.inputs.review_pack_input_csv" in review_script
    assert "--low-confidence-threshold" in review_script
    assert "--top-k" in review_script
    assert "top_review_reasons=" in review_script
    assert "top_primary_sources=" in review_script
    assert "sample_explanations=" in review_script

    sweep_step = _get_step(workflow, "evaluate", "Run Graph2D train recipe sweep (optional)")
    sweep_script = sweep_step["run"]
    assert "scripts/sweep_graph2d_train_recipes.py" in sweep_script
    assert "--recipes" in sweep_script
    assert "--seeds" in sweep_script
    assert "--base-args-json" in sweep_script

    gate_step = _get_step(workflow, "evaluate", "Check Graph2D review-pack gate (optional)")
    gate_script = gate_step["run"]
    assert "scripts/ci/check_graph2d_review_pack_gate.py" in gate_script
    assert "--summary-json" in gate_script
    assert "--config" in gate_script
    assert "--max-candidate-rate" in gate_script
    assert "--max-hybrid-rejected-rate" in gate_script

    annotation_step = _get_step(
        workflow, "evaluate", "Emit Graph2D review gate annotations (optional)"
    )
    annotation_script = annotation_step["run"]
    assert "scripts/ci/emit_graph2d_review_pack_gate_annotations.py" in annotation_script

    strict_step = _get_step(
        workflow, "evaluate", "Evaluate Graph2D review gate strict mode (optional)"
    )
    strict_script = strict_step["run"]
    assert "GRAPH2D_REVIEW_PACK_GATE_STRICT" in strict_script
    assert "review_gate_strict" in strict_script
    assert "gate status is not passed" in strict_script
    assert strict_step["continue-on-error"] == "true"

    final_fail_step = _get_step(
        workflow,
        "evaluate",
        "Fail workflow when Graph2D review gate strict check requires blocking",
    )
    assert (
        final_fail_step["if"]
        == "steps.graph2d_review_gate_strict.outputs.should_fail == 'true'"
    )
    assert "Failure reason" in final_fail_step["run"]


def test_workflow_uploads_new_graph2d_artifacts_and_summary_lines() -> None:
    workflow = _load_workflow()

    upload_review = _get_step(workflow, "evaluate", "Upload Graph2D review pack")
    assert upload_review["if"] == "steps.graph2d_review_pack.outputs.enabled == 'true'"

    upload_sweep = _get_step(workflow, "evaluate", "Upload Graph2D train sweep")
    assert upload_sweep["if"] == "steps.graph2d_train_sweep.outputs.enabled == 'true'"

    summary_step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = summary_step["run"]
    assert "Graph2D review candidates" in summary_script
    assert "Graph2D review gate status" in summary_script
    assert "Graph2D review gate headline" in summary_script
    assert "Graph2D review top reasons" in summary_script
    assert "Graph2D review top sources" in summary_script
    assert "Graph2D review example explanations" in summary_script
    assert "Graph2D review gate strict_mode" in summary_script
    assert "Graph2D train sweep total_runs" in summary_script
    assert "Graph2D train sweep best run script" in summary_script

    pr_comment_step = _get_step(workflow, "evaluate", "Comment PR with results")
    pr_comment_script = pr_comment_step["with"]["script"]
    assert "Graph2D Review Gate" in pr_comment_script
    assert "Graph2D Review Gate Strict" in pr_comment_script
    assert "Graph2D Train Sweep" in pr_comment_script
    assert "Graph2D Review Insights" in pr_comment_script
    assert "Graph2D Signal Lights" in pr_comment_script
    assert "script=${sweepBestRunScript}" in pr_comment_script
