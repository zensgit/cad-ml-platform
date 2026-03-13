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
    assert "HYBRID_CONFIDENCE_CALIBRATION_ENABLE" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_INPUT_CSV" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_OUTPUT_JSON" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_GATE_CONFIG" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_GATE_BASELINE_JSON" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_GATE_MISSING_MODE" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_FAIL_ON_GATE_FAILED" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_BASELINE_UPDATE_ENABLE" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_BASELINE_OUTPUT_JSON" in env
    assert "HYBRID_CONFIDENCE_CALIBRATION_BASELINE_SNAPSHOT_JSON" in env
    assert "HYBRID_BLIND_ENABLE" in env
    assert "HYBRID_BLIND_DXF_DIR" in env
    assert "HYBRID_BLIND_MANIFEST_CSV" in env
    assert "HYBRID_BLIND_SYNTH_MANIFEST" in env
    assert "HYBRID_BLIND_SYNTH_OUTPUT_DIR" in env
    assert "HYBRID_BLIND_MAX_FILES" in env
    assert "HYBRID_BLIND_FAMILY_PREFIX_LEN" in env
    assert "HYBRID_BLIND_FAMILY_MAP_JSON" in env
    assert "HYBRID_BLIND_FAMILY_SLICE_MAX_SNAPSHOTS" in env
    assert "HYBRID_BLIND_GATE_CONFIG" in env
    assert "HYBRID_BLIND_FAIL_ON_GATE_FAILED" in env
    assert "HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_ENABLE" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_CONSECUTIVE_WINDOW" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_ENABLE" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MIN_COMMON" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_AUTO_CAP_MIN_COMMON" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MIN_SUPPORT" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MAX_ACC_DROP" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_LABEL_SLICE_MAX_GAIN_DROP" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_ENABLE" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MIN_COMMON" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_AUTO_CAP_MIN_COMMON" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MIN_SUPPORT" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MAX_ACC_DROP" in env
    assert "HYBRID_BLIND_DRIFT_ALERT_FAMILY_SLICE_MAX_GAIN_DROP" in env

    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "review_gate_min_total_rows" in dispatch_inputs
    assert "review_gate_max_candidate_rate" in dispatch_inputs
    assert "review_gate_max_hybrid_rejected_rate" in dispatch_inputs
    assert "review_gate_max_conflict_rate" in dispatch_inputs
    assert "review_gate_max_low_confidence_rate" in dispatch_inputs
    assert "review_gate_strict" in dispatch_inputs
    assert "review_pack_input_csv" in dispatch_inputs
    assert "hybrid_calibration_enable" in dispatch_inputs
    assert "hybrid_calibration_input_csv" in dispatch_inputs
    assert "hybrid_calibration_missing_mode" in dispatch_inputs
    assert "hybrid_calibration_fail_on_gate_failed" in dispatch_inputs
    assert "hybrid_calibration_update_baseline" in dispatch_inputs
    assert "hybrid_blind_enable" in dispatch_inputs
    assert "hybrid_blind_dxf_dir" in dispatch_inputs
    assert "hybrid_blind_manifest_csv" in dispatch_inputs
    assert "hybrid_blind_synth_manifest" in dispatch_inputs
    assert "hybrid_blind_fail_on_gate_failed" in dispatch_inputs
    assert "hybrid_blind_strict_require_real_data" in dispatch_inputs
    assert "hybrid_blind_drift_alert_enable" in dispatch_inputs
    assert "hybrid_blind_drift_alert_consecutive_window" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_enable" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_min_common" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_auto_cap_min_common" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_min_support" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_max_acc_drop" in dispatch_inputs
    assert "hybrid_blind_drift_alert_label_slice_max_gain_drop" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_enable" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_min_common" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_auto_cap_min_common" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_min_support" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_max_acc_drop" in dispatch_inputs
    assert "hybrid_blind_drift_alert_family_slice_max_gain_drop" in dispatch_inputs


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

    hybrid_cal_step = _get_step(
        workflow, "evaluate", "Calibrate Hybrid confidence from review CSV (optional)"
    )
    hybrid_cal_script = hybrid_cal_step["run"]
    assert "scripts/calibrate_hybrid_confidence.py" in hybrid_cal_script
    assert "hybrid_calibration_enable" in hybrid_cal_script
    assert "--input-csv" in hybrid_cal_script
    assert "--output-json" in hybrid_cal_script

    hybrid_gate_step = _get_step(
        workflow, "evaluate", "Check Hybrid confidence calibration gate (optional)"
    )
    hybrid_gate_script = hybrid_gate_step["run"]
    assert "scripts/ci/check_hybrid_confidence_calibration_gate.py" in hybrid_gate_script
    assert "--current-json" in hybrid_gate_script
    assert "--baseline-json" in hybrid_gate_script
    assert "--config" in hybrid_gate_script
    assert "--missing-mode" in hybrid_gate_script
    assert "--output-json" in hybrid_gate_script

    hybrid_summary_step = _get_step(
        workflow,
        "evaluate",
        "Append Hybrid confidence calibration gate summary (optional)",
    )
    hybrid_summary_script = hybrid_summary_step["run"]
    assert (
        "scripts/ci/summarize_hybrid_confidence_calibration_gate.py"
        in hybrid_summary_script
    )

    hybrid_strict_step = _get_step(
        workflow, "evaluate", "Evaluate Hybrid calibration gate strict mode (optional)"
    )
    hybrid_strict_script = hybrid_strict_step["run"]
    assert "HYBRID_CONFIDENCE_CALIBRATION_FAIL_ON_GATE_FAILED" in hybrid_strict_script
    assert "hybrid_calibration_fail_on_gate_failed" in hybrid_strict_script
    assert "gate status is not passed" in hybrid_strict_script

    hybrid_baseline_step = _get_step(
        workflow, "evaluate", "Update Hybrid calibration baseline (optional)"
    )
    hybrid_baseline_script = hybrid_baseline_step["run"]
    assert (
        "scripts/ci/update_hybrid_confidence_calibration_baseline.py"
        in hybrid_baseline_script
    )
    assert "hybrid_calibration_update_baseline" in hybrid_baseline_script
    assert "HYBRID_CONFIDENCE_CALIBRATION_BASELINE_UPDATE_ENABLE" in hybrid_baseline_script
    assert "--current-json" in hybrid_baseline_script
    assert "--output-baseline-json" in hybrid_baseline_script

    hybrid_blind_eval_step = _get_step(
        workflow, "evaluate", "Run Hybrid blind benchmark (optional)"
    )
    hybrid_blind_eval_script = hybrid_blind_eval_step["run"]
    assert "scripts/batch_analyze_dxf_local.py" in hybrid_blind_eval_script
    assert "scripts/ci/build_hybrid_blind_synthetic_dxf_dataset.py" in hybrid_blind_eval_script
    assert "hybrid_blind_enable" in hybrid_blind_eval_script
    assert "hybrid_blind_synth_manifest" in hybrid_blind_eval_script
    assert "--geometry-only" in hybrid_blind_eval_script

    hybrid_blind_gate_step = _get_step(
        workflow, "evaluate", "Check Hybrid blind gate (optional)"
    )
    hybrid_blind_gate_script = hybrid_blind_gate_step["run"]
    assert "scripts/ci/check_hybrid_blind_gate.py" in hybrid_blind_gate_script
    assert "--summary-json" in hybrid_blind_gate_script
    assert "--config" in hybrid_blind_gate_script
    assert "--output" in hybrid_blind_gate_script

    hybrid_blind_strict_step = _get_step(
        workflow, "evaluate", "Evaluate Hybrid blind gate strict mode (optional)"
    )
    hybrid_blind_strict_script = hybrid_blind_strict_step["run"]
    assert "HYBRID_BLIND_FAIL_ON_GATE_FAILED" in hybrid_blind_strict_script
    assert "hybrid_blind_fail_on_gate_failed" in hybrid_blind_strict_script
    assert "HYBRID_BLIND_STRICT_REQUIRE_REAL_DATA" in hybrid_blind_strict_script
    assert "hybrid_blind_strict_require_real_data" in hybrid_blind_strict_script
    assert "strict_mode_requires_real_dataset" in hybrid_blind_strict_script
    assert "gate status is not passed" in hybrid_blind_strict_script

    hybrid_blind_history_step = _get_step(
        workflow, "evaluate", "Archive Hybrid blind eval history snapshot (optional)"
    )
    hybrid_blind_history_script = hybrid_blind_history_step["run"]
    assert "scripts/ci/archive_hybrid_blind_eval_history.py" in hybrid_blind_history_script
    assert "--summary-json" in hybrid_blind_history_script
    assert "--gate-report-json" in hybrid_blind_history_script
    assert "--output-dir reports/eval_history" in hybrid_blind_history_script
    assert "--label-slice-min-support" in hybrid_blind_history_script
    assert "--label-slice-max-slices" in hybrid_blind_history_script
    assert "--family-prefix-len" in hybrid_blind_history_script
    assert "--family-map-json" in hybrid_blind_history_script
    assert "--family-slice-max-slices" in hybrid_blind_history_script

    hybrid_blind_drift_step = _get_step(
        workflow, "evaluate", "Check Hybrid blind drift alerts (optional)"
    )
    hybrid_blind_drift_script = hybrid_blind_drift_step["run"]
    assert "scripts/ci/check_hybrid_blind_drift_alerts.py" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_enable" in hybrid_blind_drift_script
    assert "--output-md" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_consecutive_window" in hybrid_blind_drift_script
    assert "--consecutive-drop-window" in hybrid_blind_drift_script
    assert "--max-hybrid-accuracy-drop" in hybrid_blind_drift_script
    assert "--max-gain-drop" in hybrid_blind_drift_script
    assert "--max-coverage-drop" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_label_slice_enable" in hybrid_blind_drift_script
    assert "--label-slice-enable" in hybrid_blind_drift_script
    assert "--label-slice-min-common" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_label_slice_auto_cap_min_common" in hybrid_blind_drift_script
    assert "--label-slice-auto-cap-min-common" in hybrid_blind_drift_script
    assert "--no-label-slice-auto-cap-min-common" in hybrid_blind_drift_script
    assert "--label-slice-min-support" in hybrid_blind_drift_script
    assert "--label-slice-max-hybrid-accuracy-drop" in hybrid_blind_drift_script
    assert "--label-slice-max-gain-drop" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_family_slice_enable" in hybrid_blind_drift_script
    assert "--family-slice-enable" in hybrid_blind_drift_script
    assert "--family-slice-min-common" in hybrid_blind_drift_script
    assert "hybrid_blind_drift_alert_family_slice_auto_cap_min_common" in hybrid_blind_drift_script
    assert "--family-slice-auto-cap-min-common" in hybrid_blind_drift_script
    assert "--no-family-slice-auto-cap-min-common" in hybrid_blind_drift_script
    assert "--family-slice-min-support" in hybrid_blind_drift_script
    assert "--family-slice-max-hybrid-accuracy-drop" in hybrid_blind_drift_script
    assert "--family-slice-max-gain-drop" in hybrid_blind_drift_script

    hybrid_final_fail_step = _get_step(
        workflow,
        "evaluate",
        "Fail workflow when Hybrid calibration gate strict check requires blocking",
    )
    assert (
        hybrid_final_fail_step["if"]
        == "steps.hybrid_calibration_gate_strict.outputs.should_fail == 'true'"
    )
    assert "Failure reason" in hybrid_final_fail_step["run"]

    hybrid_blind_final_fail_step = _get_step(
        workflow,
        "evaluate",
        "Fail workflow when Hybrid blind gate strict check requires blocking",
    )
    assert (
        hybrid_blind_final_fail_step["if"]
        == "steps.hybrid_blind_gate_strict.outputs.should_fail == 'true'"
    )
    assert "Failure reason" in hybrid_blind_final_fail_step["run"]


def test_workflow_uploads_new_graph2d_artifacts_and_summary_lines() -> None:
    workflow = _load_workflow()

    upload_review = _get_step(workflow, "evaluate", "Upload Graph2D review pack")
    assert upload_review["if"] == "steps.graph2d_review_pack.outputs.enabled == 'true'"

    upload_sweep = _get_step(workflow, "evaluate", "Upload Graph2D train sweep")
    assert upload_sweep["if"] == "steps.graph2d_train_sweep.outputs.enabled == 'true'"

    upload_hybrid = _get_step(
        workflow, "evaluate", "Upload Hybrid confidence calibration artifacts"
    )
    assert upload_hybrid["if"] == "steps.hybrid_calibration.outputs.enabled == 'true'"

    upload_hybrid_baseline = _get_step(
        workflow, "evaluate", "Upload Hybrid confidence calibration baseline artifact"
    )
    assert (
        upload_hybrid_baseline["if"]
        == "steps.hybrid_calibration_baseline_update.outputs.enabled == 'true'"
    )

    upload_hybrid_blind = _get_step(
        workflow, "evaluate", "Upload Hybrid blind benchmark artifacts"
    )
    assert upload_hybrid_blind["if"] == "steps.hybrid_blind_eval.outputs.enabled == 'true'"

    upload_hybrid_blind_history = _get_step(
        workflow, "evaluate", "Upload Hybrid blind history snapshot"
    )
    assert (
        upload_hybrid_blind_history["if"]
        == "steps.hybrid_blind_history.outputs.enabled == 'true'"
    )

    upload_hybrid_blind_drift = _get_step(
        workflow, "evaluate", "Upload Hybrid blind drift alert report"
    )
    assert (
        upload_hybrid_blind_drift["if"]
        == "steps.hybrid_blind_drift_alert.outputs.enabled == 'true'"
    )
    assert "report_md_path" in upload_hybrid_blind_drift["with"]["path"]

    weekly_summary_step = _get_step(workflow, "evaluate", "Generate weekly rolling summary")
    weekly_summary_script = weekly_summary_step["run"]
    assert "scripts/ci/generate_eval_weekly_summary.py" in weekly_summary_script
    assert "--eval-history-dir" in weekly_summary_script
    assert "--output-md" in weekly_summary_script
    assert "--days 7" in weekly_summary_script

    upload_weekly_summary = _get_step(workflow, "evaluate", "Upload weekly summary")
    assert upload_weekly_summary["with"]["name"] == "evaluation-weekly-summary-${{ github.run_number }}"

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
    assert "Hybrid calibration status" in summary_script
    assert "Hybrid calibration metrics_after" in summary_script
    assert "Hybrid calibration gate status" in summary_script
    assert "Hybrid calibration gate strict_should_fail" in summary_script
    assert "Hybrid calibration baseline update" in summary_script
    assert "Hybrid calibration baseline path" in summary_script
    assert "Weekly summary" in summary_script
    assert "Hybrid blind eval status" in summary_script
    assert "Hybrid blind dataset_source" in summary_script
    assert "Hybrid blind dxf_dir" in summary_script
    assert "Hybrid blind gain_vs_graph2d" in summary_script
    assert "Hybrid blind gate status" in summary_script
    assert "Hybrid blind strict_require_real_data" in summary_script
    assert "Hybrid blind gate strict_should_fail" in summary_script
    assert "Hybrid blind history snapshot" in summary_script
    assert "Hybrid blind drift alert status" in summary_script
    assert "Hybrid blind drift report" in summary_script
    assert "Hybrid blind delta_hybrid_accuracy" in summary_script
    assert "Hybrid blind label_slice_enabled" in summary_script
    assert "Hybrid blind label_slice_auto_cap_min_common" in summary_script
    assert "Hybrid blind label_slice_effective_min_common" in summary_script
    assert "Hybrid blind label_slice_common_count" in summary_script
    assert "Hybrid blind label_slice_worst_acc_drop" in summary_script
    assert "Hybrid blind family_slice_enabled" in summary_script
    assert "Hybrid blind family_slice_auto_cap_min_common" in summary_script
    assert "Hybrid blind family_slice_effective_min_common" in summary_script
    assert "Hybrid blind family_slice_common_count" in summary_script
    assert "Hybrid blind family_slice_worst_acc_drop" in summary_script

    pr_comment_step = _get_step(workflow, "evaluate", "Comment PR with results")
    pr_comment_script = pr_comment_step["with"]["script"]
    assert "Graph2D Review Gate" in pr_comment_script
    assert "Graph2D Review Gate Strict" in pr_comment_script
    assert "Graph2D Train Sweep" in pr_comment_script
    assert "Graph2D Review Insights" in pr_comment_script
    assert "Hybrid Calibration" in pr_comment_script
    assert "Hybrid Calibration Gate" in pr_comment_script
    assert "Hybrid Calibration Strict" in pr_comment_script
    assert "Hybrid Calibration Baseline" in pr_comment_script
    assert "Hybrid Blind Eval" in pr_comment_script
    assert "Hybrid Blind Gate" in pr_comment_script
    assert "Hybrid Blind Strict" in pr_comment_script
    assert "Hybrid Blind Drift Alert" in pr_comment_script
    assert "source=${hybridBlindDatasetSource" in pr_comment_script
    assert "require_real=${hybridBlindStrictRequireReal" in pr_comment_script
    assert "label_slice_enabled=${hybridBlindLabelSliceEnabled" in pr_comment_script
    assert "label_auto_cap=${hybridBlindLabelSliceAutoCap" in pr_comment_script
    assert "label_effective_min_common=${hybridBlindLabelSliceEffectiveMinCommon" in pr_comment_script
    assert "worst_label_acc_drop=${hybridBlindLabelSliceWorstAccDrop" in pr_comment_script
    assert "family_slice_enabled=${hybridBlindFamilySliceEnabled" in pr_comment_script
    assert "family_auto_cap=${hybridBlindFamilySliceAutoCap" in pr_comment_script
    assert "family_effective_min_common=${hybridBlindFamilySliceEffectiveMinCommon" in pr_comment_script
    assert "worst_family_acc_drop=${hybridBlindFamilySliceWorstAccDrop" in pr_comment_script
    assert "Blind Gain (Hybrid-Graph2D)" in pr_comment_script
    assert "| **Hybrid Blind** | ${hybridBlindLight}" in pr_comment_script
    assert "Signal Lights" in pr_comment_script
    assert "script=${sweepBestRunScript}" in pr_comment_script


def test_workflow_runs_eval_with_history_regression_tests() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Run eval_with_history regression unit tests")
    run_script = step["run"]
    assert "pytest -q" in run_script
    assert "tests/unit/test_eval_with_history_script_history_sequence.py" in run_script
    assert "tests/unit/test_validate_eval_history_history_sequence.py" in run_script


def test_workflow_runs_hybrid_calibration_regression_tests() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Run hybrid calibration regression unit tests")
    run_script = step["run"]
    assert "pytest -q" in run_script
    assert "tests/unit/test_calibrate_hybrid_confidence_script.py" in run_script
    assert "tests/unit/test_hybrid_confidence_calibration_gate_check.py" in run_script
