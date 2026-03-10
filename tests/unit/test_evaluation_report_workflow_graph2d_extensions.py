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
    permissions = workflow["jobs"]["evaluate"]["permissions"]

    assert "GRAPH2D_REVIEW_PACK_INPUT_CSV" in env
    assert "GRAPH2D_REVIEW_PACK_INPUT_ARTIFACT_NAME" in env
    assert "GRAPH2D_REVIEW_PACK_INPUT_ARTIFACT_RUN_ID" in env
    assert "GRAPH2D_REVIEW_PACK_INPUT_ARTIFACT_REPOSITORY" in env
    assert "GRAPH2D_REVIEW_PACK_INPUT_ARTIFACT_PATH" in env
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
    assert "BENCHMARK_SCORECARD_ENABLE" in env
    assert "BENCHMARK_SCORECARD_TITLE" in env
    assert "BENCHMARK_SCORECARD_HYBRID_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_GRAPH2D_METRICS_JSON" in env
    assert "BENCHMARK_SCORECARD_GRAPH2D_DIAGNOSE_JSON" in env
    assert "BENCHMARK_SCORECARD_GRAPH2D_BLIND_DIAGNOSE_JSON" in env
    assert "BENCHMARK_SCORECARD_HISTORY_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_BREP_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_MIGRATION_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_ASSISTANT_EVIDENCE_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_FEEDBACK_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_FINETUNE_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_METRIC_TRAIN_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_OCR_REVIEW_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_QDRANT_READINESS_JSON" in env
    assert "BENCHMARK_SCORECARD_ENGINEERING_SIGNALS_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_SCORECARD_OPERATOR_ADOPTION_SUMMARY_JSON" in env
    assert "BENCHMARK_SCORECARD_OUTPUT_JSON" in env
    assert "BENCHMARK_SCORECARD_OUTPUT_MD" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_ENABLE" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_TITLE" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_HYBRID_SUMMARY_JSON" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_OCR_REVIEW_SUMMARY_JSON" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_JSON" in env
    assert "BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_MD" in env
    assert "BENCHMARK_REALDATA_SIGNALS_ENABLE" in env
    assert "BENCHMARK_REALDATA_SIGNALS_TITLE" in env
    assert "BENCHMARK_REALDATA_SIGNALS_HYBRID_SUMMARY_JSON" in env
    assert "BENCHMARK_REALDATA_SIGNALS_ONLINE_EXAMPLE_REPORT_JSON" in env
    assert "BENCHMARK_REALDATA_SIGNALS_STEP_DIR_SUMMARY_JSON" in env
    assert "BENCHMARK_REALDATA_SIGNALS_OUTPUT_JSON" in env
    assert "BENCHMARK_REALDATA_SIGNALS_OUTPUT_MD" in env
    assert "BENCHMARK_REALDATA_SCORECARD_ENABLE" in env
    assert "BENCHMARK_REALDATA_SCORECARD_TITLE" in env
    assert "BENCHMARK_REALDATA_SCORECARD_HYBRID_SUMMARY_JSON" in env
    assert "BENCHMARK_REALDATA_SCORECARD_HISTORY_SUMMARY_JSON" in env
    assert "BENCHMARK_REALDATA_SCORECARD_ONLINE_EXAMPLE_REPORT_JSON" in env
    assert "BENCHMARK_REALDATA_SCORECARD_STEP_DIR_SUMMARY_JSON" in env
    assert "BENCHMARK_REALDATA_SCORECARD_OUTPUT_JSON" in env
    assert "BENCHMARK_REALDATA_SCORECARD_OUTPUT_MD" in env
    assert "BENCHMARK_KNOWLEDGE_READINESS_ENABLE" in env
    assert "BENCHMARK_KNOWLEDGE_READINESS_TITLE" in env
    assert "BENCHMARK_KNOWLEDGE_READINESS_SNAPSHOT_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_MD" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_ENABLE" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_TITLE" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_MD" in env
    assert "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ENABLE" in env
    assert "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_TITLE" in env
    assert "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_KNOWLEDGE_DOMAIN_MATRIX_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_MD" in env
    assert "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_ENABLE" in env
    assert "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_TITLE" in env
    assert "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_KNOWLEDGE_SOURCE_COVERAGE_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_JSON" in env
    assert "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_MD" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_ENABLE" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_TITLE" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_APPLICATION_JSON" in env
    assert (
        "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_REALDATA_CORRELATION_JSON"
        in env
    )
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_DOMAIN_MATRIX_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON" in env
    assert (
        "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_OUTCOME_CORRELATION_JSON"
        in env
    )
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_KNOWLEDGE_OUTCOME_DRIFT_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_REALDATA_SIGNALS_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_REALDATA_SCORECARD_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_OPERATOR_ADOPTION_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_JSON" in env
    assert "BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_MD" in env
    assert "FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_JSON" in env
    assert "FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_MD" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_ENABLE" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_TITLE" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_SCORECARD_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_FEEDBACK_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_ASSISTANT_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_REVIEW_QUEUE_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_OCR_REVIEW_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_JSON" in env
    assert "BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_MD" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_ENABLE" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_TITLE" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_SUMMARY_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_OCR_REVIEW_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_COMPANION_SUMMARY_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_RELEASE_DECISION_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SIGNALS_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_JSON" in env
    assert "BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_MD" in env
    assert "BENCHMARK_COMPANION_SUMMARY_ENABLE" in env
    assert "BENCHMARK_COMPANION_SUMMARY_TITLE" in env
    assert "BENCHMARK_COMPANION_SUMMARY_SCORECARD_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_SUMMARY_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_ARTIFACT_BUNDLE_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_REALDATA_SIGNALS_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_INDEX_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_OUTPUT_JSON" in env
    assert "BENCHMARK_COMPANION_SUMMARY_OUTPUT_MD" in env
    assert "BENCHMARK_RELEASE_DECISION_ENABLE" in env
    assert "BENCHMARK_RELEASE_DECISION_TITLE" in env
    assert "BENCHMARK_RELEASE_DECISION_SCORECARD_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_OPERATIONAL_SUMMARY_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_ARTIFACT_BUNDLE_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_COMPANION_SUMMARY_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_REALDATA_SIGNALS_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_INDEX_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_OUTPUT_JSON" in env
    assert "BENCHMARK_RELEASE_DECISION_OUTPUT_MD" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_ENABLE" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_TITLE" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_RELEASE_DECISION_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_COMPANION_SUMMARY_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_ARTIFACT_BUNDLE_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_ENGINEERING_SIGNALS_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_REALDATA_SIGNALS_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_READINESS_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_INDEX_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_OUTPUT_JSON" in env
    assert "BENCHMARK_RELEASE_RUNBOOK_OUTPUT_MD" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_ENABLE" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_TITLE" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_RELEASE_DECISION_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_RELEASE_RUNBOOK_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_REVIEW_QUEUE_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_FEEDBACK_FLYWHEEL_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_OUTPUT_JSON" in env
    assert "BENCHMARK_OPERATOR_ADOPTION_OUTPUT_MD" in env
    assert "OCR_REVIEW_PACK_ENABLE" in env
    assert "OCR_REVIEW_PACK_INPUT" in env
    assert "OCR_REVIEW_PACK_OUTPUT_CSV" in env
    assert "OCR_REVIEW_PACK_OUTPUT_JSON" in env
    assert "OCR_REVIEW_PACK_TOP_K" in env
    assert "OCR_REVIEW_PACK_INCLUDE_READY" in env
    assert "ASSISTANT_EVIDENCE_REPORT_ENABLE" in env
    assert "ASSISTANT_EVIDENCE_REPORT_INPUT" in env
    assert "ASSISTANT_EVIDENCE_REPORT_OUTPUT_CSV" in env
    assert "ASSISTANT_EVIDENCE_REPORT_OUTPUT_JSON" in env
    assert "ASSISTANT_EVIDENCE_REPORT_TOP_K" in env
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLE" in env
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_INPUT" in env
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_CSV" in env
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON" in env
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_K" in env

    dispatch_inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "review_gate_min_total_rows" in dispatch_inputs
    assert "review_gate_max_candidate_rate" in dispatch_inputs
    assert "review_gate_max_hybrid_rejected_rate" in dispatch_inputs
    assert "review_gate_max_conflict_rate" in dispatch_inputs
    assert "review_gate_max_low_confidence_rate" in dispatch_inputs
    assert "review_gate_strict" in dispatch_inputs
    assert "review_pack_input_csv" in dispatch_inputs
    assert "review_pack_input_artifact_name" in dispatch_inputs
    assert "review_pack_input_artifact_run_id" in dispatch_inputs
    assert "review_pack_input_artifact_repository" in dispatch_inputs
    assert "review_pack_input_artifact_path" in dispatch_inputs
    assert "benchmark_scorecard_enable" in dispatch_inputs
    assert "benchmark_scorecard_assistant_evidence_summary" in dispatch_inputs
    assert "benchmark_scorecard_review_queue_summary" in dispatch_inputs
    assert "benchmark_scorecard_feedback_summary" in dispatch_inputs
    assert "benchmark_scorecard_finetune_summary" in dispatch_inputs
    assert "benchmark_scorecard_metric_train_summary" in dispatch_inputs
    assert "benchmark_scorecard_ocr_review_summary" in dispatch_inputs
    assert "benchmark_scorecard_qdrant_readiness_summary" in dispatch_inputs
    assert "benchmark_scorecard_engineering_signals_summary" in dispatch_inputs
    assert "benchmark_scorecard_knowledge_readiness_summary" in dispatch_inputs
    assert "benchmark_scorecard_operator_adoption_summary" in dispatch_inputs
    assert "benchmark_engineering_signals_enable" in dispatch_inputs
    assert "benchmark_engineering_signals_hybrid_summary_json" in dispatch_inputs
    assert "benchmark_engineering_signals_ocr_review_summary_json" in dispatch_inputs
    assert "benchmark_realdata_signals_enable" in dispatch_inputs
    assert "benchmark_realdata_signals_hybrid_summary_json" in dispatch_inputs
    assert "benchmark_realdata_signals_online_example_report_json" in dispatch_inputs
    assert "benchmark_realdata_signals_step_dir_summary_json" in dispatch_inputs
    assert "benchmark_realdata_scorecard_enable" in dispatch_inputs
    assert "benchmark_realdata_scorecard_hybrid_summary_json" in dispatch_inputs
    assert "benchmark_realdata_scorecard_history_summary_json" in dispatch_inputs
    assert "benchmark_realdata_scorecard_online_example_report_json" in dispatch_inputs
    assert "benchmark_realdata_scorecard_step_dir_summary_json" in dispatch_inputs
    assert "benchmark_knowledge_readiness_enable" in dispatch_inputs
    assert "benchmark_knowledge_readiness_snapshot_json" in dispatch_inputs
    assert "benchmark_knowledge_application_enable" in dispatch_inputs
    assert "benchmark_knowledge_application_engineering_signals_json" in dispatch_inputs
    assert "benchmark_knowledge_application_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_knowledge_domain_action_plan_enable" in dispatch_inputs
    assert "benchmark_knowledge_domain_action_plan_knowledge_domain_matrix_json" in dispatch_inputs
    assert "benchmark_knowledge_source_action_plan_enable" in dispatch_inputs
    assert (
        "benchmark_knowledge_source_action_plan_knowledge_source_coverage_json"
        in dispatch_inputs
    )
    assert "benchmark_knowledge_outcome_correlation_enable" in dispatch_inputs
    assert (
        "benchmark_knowledge_outcome_correlation_knowledge_domain_matrix_json"
        in dispatch_inputs
    )
    assert "benchmark_knowledge_outcome_correlation_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_knowledge_outcome_drift_enable" in dispatch_inputs
    assert "benchmark_knowledge_outcome_drift_current_summary_json" in dispatch_inputs
    assert "benchmark_knowledge_outcome_drift_previous_summary_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_enable" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_engineering_signals_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_knowledge_application_json" in dispatch_inputs
    assert (
        "benchmark_competitive_surpass_index_knowledge_realdata_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_competitive_surpass_index_knowledge_domain_matrix_json" in dispatch_inputs
    assert (
        "benchmark_competitive_surpass_index_knowledge_domain_action_plan_json"
        in dispatch_inputs
    )
    assert (
        "benchmark_competitive_surpass_index_knowledge_source_action_plan_json"
        in dispatch_inputs
    )
    assert (
        "benchmark_competitive_surpass_index_knowledge_outcome_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_competitive_surpass_index_knowledge_outcome_drift_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_realdata_signals_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_competitive_surpass_index_operator_adoption_json" in dispatch_inputs
    assert "benchmark_operational_summary_enable" in dispatch_inputs
    assert "benchmark_operational_summary_scorecard_json" in dispatch_inputs
    assert "benchmark_operational_summary_feedback_json" in dispatch_inputs
    assert "benchmark_operational_summary_assistant_json" in dispatch_inputs
    assert "benchmark_operational_summary_review_queue_json" in dispatch_inputs
    assert "benchmark_operational_summary_ocr_review_json" in dispatch_inputs
    assert "benchmark_operational_summary_operator_adoption_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_enable" in dispatch_inputs
    assert "benchmark_artifact_bundle_scorecard_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_operational_summary_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_feedback_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_assistant_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_review_queue_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_ocr_review_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_companion_summary_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_competitive_surpass_index_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_release_decision_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_engineering_signals_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_realdata_signals_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_knowledge_drift_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_knowledge_application_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_knowledge_domain_action_plan_json" in dispatch_inputs
    assert "benchmark_artifact_bundle_knowledge_source_action_plan_json" in dispatch_inputs
    assert (
        "benchmark_artifact_bundle_knowledge_outcome_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_artifact_bundle_knowledge_outcome_drift_json" in dispatch_inputs
    assert "benchmark_companion_summary_enable" in dispatch_inputs
    assert "benchmark_companion_summary_scorecard_json" in dispatch_inputs
    assert "benchmark_companion_summary_operational_summary_json" in dispatch_inputs
    assert "benchmark_companion_summary_artifact_bundle_json" in dispatch_inputs
    assert "benchmark_companion_summary_engineering_signals_json" in dispatch_inputs
    assert "benchmark_companion_summary_realdata_signals_json" in dispatch_inputs
    assert "benchmark_companion_summary_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_companion_summary_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_companion_summary_knowledge_drift_json" in dispatch_inputs
    assert "benchmark_companion_summary_knowledge_application_json" in dispatch_inputs
    assert "benchmark_companion_summary_knowledge_domain_action_plan_json" in dispatch_inputs
    assert "benchmark_companion_summary_knowledge_source_action_plan_json" in dispatch_inputs
    assert "benchmark_companion_summary_competitive_surpass_index_json" in dispatch_inputs
    assert (
        "benchmark_companion_summary_knowledge_outcome_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_companion_summary_knowledge_outcome_drift_json" in dispatch_inputs
    assert "benchmark_release_decision_enable" in dispatch_inputs
    assert "benchmark_release_decision_scorecard_json" in dispatch_inputs
    assert "benchmark_release_decision_operational_summary_json" in dispatch_inputs
    assert "benchmark_release_decision_artifact_bundle_json" in dispatch_inputs
    assert "benchmark_release_decision_companion_summary_json" in dispatch_inputs
    assert "benchmark_release_decision_engineering_signals_json" in dispatch_inputs
    assert "benchmark_release_decision_realdata_signals_json" in dispatch_inputs
    assert "benchmark_release_decision_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_release_decision_operator_adoption_json" in dispatch_inputs
    assert "benchmark_release_decision_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_release_decision_knowledge_drift_json" in dispatch_inputs
    assert "benchmark_release_decision_knowledge_application_json" in dispatch_inputs
    assert "benchmark_release_decision_knowledge_domain_action_plan_json" in dispatch_inputs
    assert "benchmark_release_decision_knowledge_source_action_plan_json" in dispatch_inputs
    assert (
        "benchmark_release_decision_knowledge_outcome_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_release_decision_knowledge_outcome_drift_json" in dispatch_inputs
    assert "benchmark_release_decision_competitive_surpass_index_json" in dispatch_inputs
    assert "benchmark_release_runbook_enable" in dispatch_inputs
    assert "benchmark_release_runbook_release_decision_json" in dispatch_inputs
    assert "benchmark_release_runbook_companion_summary_json" in dispatch_inputs
    assert "benchmark_release_runbook_artifact_bundle_json" in dispatch_inputs
    assert "benchmark_release_runbook_engineering_signals_json" in dispatch_inputs
    assert "benchmark_release_runbook_realdata_signals_json" in dispatch_inputs
    assert "benchmark_release_runbook_realdata_scorecard_json" in dispatch_inputs
    assert "benchmark_release_runbook_operator_adoption_json" in dispatch_inputs
    assert "benchmark_release_runbook_knowledge_readiness_json" in dispatch_inputs
    assert "benchmark_release_runbook_knowledge_drift_json" in dispatch_inputs
    assert "benchmark_release_runbook_knowledge_application_json" in dispatch_inputs
    assert "benchmark_release_runbook_knowledge_domain_action_plan_json" in dispatch_inputs
    assert "benchmark_release_runbook_knowledge_source_action_plan_json" in dispatch_inputs
    assert (
        "benchmark_release_runbook_knowledge_outcome_correlation_json"
        in dispatch_inputs
    )
    assert "benchmark_release_runbook_knowledge_outcome_drift_json" in dispatch_inputs
    assert "benchmark_release_runbook_competitive_surpass_index_json" in dispatch_inputs
    assert "benchmark_operator_adoption_enable" in dispatch_inputs
    assert "benchmark_operator_adoption_release_decision_json" in dispatch_inputs
    assert "benchmark_operator_adoption_release_runbook_json" in dispatch_inputs
    assert "benchmark_operator_adoption_review_queue_json" in dispatch_inputs
    assert "benchmark_operator_adoption_feedback_flywheel_json" in dispatch_inputs
    assert "benchmark_operator_adoption_knowledge_drift_json" in dispatch_inputs
    assert "benchmark_operator_adoption_knowledge_outcome_drift_json" in dispatch_inputs
    assert "ocr_review_pack_enable" in dispatch_inputs
    assert "ocr_review_pack_input" in dispatch_inputs
    assert "assistant_evidence_report_enable" in dispatch_inputs
    assert "assistant_evidence_report_input" in dispatch_inputs
    assert "active_learning_review_queue_report_enable" in dispatch_inputs
    assert "active_learning_review_queue_report_input" in dispatch_inputs
    assert "active_learning_review_queue_report_top_k" in dispatch_inputs
    assert permissions["actions"] == "read"


def test_workflow_has_optional_graph2d_review_pack_and_train_sweep_steps() -> None:
    workflow = _load_workflow()
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    download_step = _get_step(
        workflow, "evaluate", "Download review-pack input artifact (optional)"
    )
    assert download_step["uses"] == "actions/download-artifact@v4"
    assert "review_pack_input_artifact_name" in download_step["if"]
    download_with = download_step["with"]
    assert "run-id" in download_with
    assert "repository" in download_with
    assert "github-token" in download_with

    review_step = _get_step(workflow, "evaluate", "Build hybrid rejection review pack (optional)")
    review_script = review_step["run"]
    assert "scripts/export_hybrid_rejection_review_pack.py" in review_script
    assert "github.event.inputs.review_pack_input_csv" in review_script
    assert "review_pack_input_artifact_path" in review_script
    assert "find \"$ARTIFACT_INPUT_DIR\" -type f -name '*.csv'" in review_script
    assert "input_source=" in review_script
    assert "--low-confidence-threshold" in review_script
    assert "--top-k" in review_script
    assert "top_review_reasons=" in review_script
    assert "top_coarse_labels=" in review_script
    assert "top_fine_labels=" in review_script
    assert "top_rejection_reasons=" in review_script
    assert "knowledge_conflict_count=" in review_script
    assert "knowledge_check_row_count=" in review_script
    assert "standards_candidate_row_count=" in review_script
    assert "top_knowledge_conflicts=" in review_script
    assert "top_review_priorities=" in review_script
    assert "top_confidence_bands=" in review_script
    assert "top_knowledge_check_categories=" in review_script
    assert "top_standard_candidate_types=" in review_script
    assert "top_knowledge_hint_labels=" in review_script
    assert "top_primary_sources=" in review_script
    assert "top_shadow_sources=" in review_script
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

    ocr_review_step = _get_step(workflow, "evaluate", "Build OCR review pack (optional)")
    ocr_review_script = ocr_review_step["run"]
    assert "scripts/export_ocr_review_pack.py" in ocr_review_script
    assert "OCR_REVIEW_PACK_ENABLE" in ocr_review_script
    assert "ocr_review_pack_input" in ocr_review_script
    assert "review_priority_counts=" in ocr_review_script
    assert "top_recommended_actions=" in ocr_review_script

    benchmark_engineering_step = _get_step(
        workflow, "evaluate", "Build benchmark engineering signals (optional)"
    )
    benchmark_engineering_script = benchmark_engineering_step["run"]
    assert "scripts/export_benchmark_engineering_signals.py" in benchmark_engineering_script
    assert "BENCHMARK_ENGINEERING_SIGNALS_ENABLE" in benchmark_engineering_script
    assert "benchmark_engineering_signals_hybrid_summary_json" in benchmark_engineering_script
    assert "benchmark_engineering_signals_ocr_review_summary_json" in benchmark_engineering_script
    assert "steps.ocr_review_pack.outputs.output_json" in benchmark_engineering_script
    assert "INPUT_COUNT=0" in benchmark_engineering_script
    assert "coverage_ratio=" in benchmark_engineering_script
    assert "rows_with_violations=" in benchmark_engineering_script
    assert "rows_with_standards_candidates=" in benchmark_engineering_script
    assert "ocr_standard_signal_count=" in benchmark_engineering_script
    assert "recommendations=" in benchmark_engineering_script

    benchmark_realdata_step = _get_step(
        workflow, "evaluate", "Build benchmark realdata signals (optional)"
    )
    benchmark_realdata_script = benchmark_realdata_step["run"]
    assert "scripts/export_benchmark_realdata_signals.py" in benchmark_realdata_script
    assert "BENCHMARK_REALDATA_SIGNALS_ENABLE" in benchmark_realdata_script
    assert "benchmark_realdata_signals_hybrid_summary_json" in benchmark_realdata_script
    assert (
        "benchmark_realdata_signals_online_example_report_json"
        in benchmark_realdata_script
    )
    assert "benchmark_realdata_signals_step_dir_summary_json" in benchmark_realdata_script
    assert "BENCHMARK_REALDATA_SIGNALS_HYBRID_SUMMARY_JSON" in benchmark_realdata_script
    assert (
        "BENCHMARK_REALDATA_SIGNALS_ONLINE_EXAMPLE_REPORT_JSON"
        in benchmark_realdata_script
    )
    assert "BENCHMARK_REALDATA_SIGNALS_STEP_DIR_SUMMARY_JSON" in benchmark_realdata_script
    assert "ready_component_count=" in benchmark_realdata_script
    assert "partial_component_count=" in benchmark_realdata_script
    assert "environment_blocked_count=" in benchmark_realdata_script
    assert "available_component_count=" in benchmark_realdata_script
    assert "hybrid_dxf_status=" in benchmark_realdata_script
    assert "history_h5_status=" in benchmark_realdata_script
    assert "step_smoke_status=" in benchmark_realdata_script
    assert "step_dir_status=" in benchmark_realdata_script
    assert "recommendations=" in benchmark_realdata_script

    benchmark_realdata_scorecard_step = _get_step(
        workflow, "evaluate", "Build benchmark real-data scorecard (optional)"
    )
    benchmark_realdata_scorecard_script = benchmark_realdata_scorecard_step["run"]
    assert (
        "scripts/export_benchmark_realdata_scorecard.py"
        in benchmark_realdata_scorecard_script
    )
    assert "BENCHMARK_REALDATA_SCORECARD_ENABLE" in benchmark_realdata_scorecard_script
    assert (
        "benchmark_realdata_scorecard_hybrid_summary_json"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "benchmark_realdata_scorecard_history_summary_json"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "benchmark_realdata_scorecard_online_example_report_json"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "benchmark_realdata_scorecard_step_dir_summary_json"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "BENCHMARK_REALDATA_SCORECARD_HYBRID_SUMMARY_JSON"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "BENCHMARK_REALDATA_SCORECARD_HISTORY_SUMMARY_JSON"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "BENCHMARK_REALDATA_SCORECARD_ONLINE_EXAMPLE_REPORT_JSON"
        in benchmark_realdata_scorecard_script
    )
    assert (
        "BENCHMARK_REALDATA_SCORECARD_STEP_DIR_SUMMARY_JSON"
        in benchmark_realdata_scorecard_script
    )
    assert "BENCHMARK_SCORECARD_HISTORY_SUMMARY_JSON" in benchmark_realdata_scorecard_script
    assert "ready_component_count=" in benchmark_realdata_scorecard_script
    assert "partial_component_count=" in benchmark_realdata_scorecard_script
    assert "environment_blocked_count=" in benchmark_realdata_scorecard_script
    assert "available_component_count=" in benchmark_realdata_scorecard_script
    assert "best_surface=" in benchmark_realdata_scorecard_script
    assert "hybrid_dxf_status=" in benchmark_realdata_scorecard_script
    assert "history_h5_status=" in benchmark_realdata_scorecard_script
    assert "step_smoke_status=" in benchmark_realdata_scorecard_script
    assert "step_dir_status=" in benchmark_realdata_scorecard_script
    assert "recommendations=" in benchmark_realdata_scorecard_script

    benchmark_knowledge_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge readiness (optional)"
    )
    benchmark_knowledge_script = benchmark_knowledge_step["run"]
    assert "scripts/export_benchmark_knowledge_readiness.py" in benchmark_knowledge_script
    assert "BENCHMARK_KNOWLEDGE_READINESS_ENABLE" in benchmark_knowledge_script
    assert "benchmark_knowledge_readiness_snapshot_json" in benchmark_knowledge_script
    assert "BENCHMARK_KNOWLEDGE_READINESS_SNAPSHOT_JSON" in benchmark_knowledge_script
    assert "total_reference_items=" in benchmark_knowledge_script
    assert "ready_component_count=" in benchmark_knowledge_script
    assert "partial_component_count=" in benchmark_knowledge_script
    assert "missing_component_count=" in benchmark_knowledge_script
    assert "focus_area_count=" in benchmark_knowledge_script
    assert "focus_areas=" in benchmark_knowledge_script
    assert "domain_count=" in benchmark_knowledge_script
    assert "priority_domains=" in benchmark_knowledge_script
    assert "domain_focus_areas=" in benchmark_knowledge_script
    assert "recommendations=" in benchmark_knowledge_script
    benchmark_knowledge_drift_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge drift (optional)"
    )
    benchmark_knowledge_drift_script = benchmark_knowledge_drift_step["run"]
    assert "scripts/export_benchmark_knowledge_drift.py" in benchmark_knowledge_drift_script
    assert "BENCHMARK_KNOWLEDGE_DRIFT_ENABLE" in benchmark_knowledge_drift_script
    assert "benchmark_knowledge_drift_current_summary_json" in benchmark_knowledge_drift_script
    assert "reference_item_delta=" in benchmark_knowledge_drift_script
    assert "regressions=" in benchmark_knowledge_drift_script
    assert "improvements=" in benchmark_knowledge_drift_script
    assert "domain_regressions=" in benchmark_knowledge_drift_script
    assert "domain_improvements=" in benchmark_knowledge_drift_script
    assert "resolved_focus_areas=" in benchmark_knowledge_drift_script
    assert "new_focus_areas=" in benchmark_knowledge_drift_script
    assert "resolved_priority_domains=" in benchmark_knowledge_drift_script
    assert "new_priority_domains=" in benchmark_knowledge_drift_script

    benchmark_knowledge_application_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge application (optional)"
    )
    benchmark_knowledge_application_script = benchmark_knowledge_application_step["run"]
    assert (
        "scripts/export_benchmark_knowledge_application.py"
        in benchmark_knowledge_application_script
    )
    assert "BENCHMARK_KNOWLEDGE_APPLICATION_ENABLE" in benchmark_knowledge_application_script
    assert (
        "benchmark_knowledge_application_engineering_signals_json"
        in benchmark_knowledge_application_script
    )
    assert (
        "benchmark_knowledge_application_knowledge_readiness_json"
        in benchmark_knowledge_application_script
    )
    assert (
        "steps.benchmark_engineering_signals.outputs.output_json"
        in benchmark_knowledge_application_script
    )
    assert (
        "steps.benchmark_knowledge_readiness.outputs.output_json"
        in benchmark_knowledge_application_script
    )
    assert "ready_domain_count=" in benchmark_knowledge_application_script
    assert "partial_domain_count=" in benchmark_knowledge_application_script
    assert "missing_domain_count=" in benchmark_knowledge_application_script
    assert "total_domain_count=" in benchmark_knowledge_application_script
    assert "focus_area_count=" in benchmark_knowledge_application_script
    assert "focus_areas=" in benchmark_knowledge_application_script
    assert "priority_domains=" in benchmark_knowledge_application_script
    assert "domain_statuses=" in benchmark_knowledge_application_script
    assert "recommendations=" in benchmark_knowledge_application_script

    benchmark_knowledge_realdata_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge realdata correlation (optional)"
    )
    benchmark_knowledge_realdata_script = benchmark_knowledge_realdata_step["run"]
    assert (
        "scripts/export_benchmark_knowledge_realdata_correlation.py"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_ENABLE"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "benchmark_knowledge_realdata_correlation_knowledge_readiness_json"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "benchmark_knowledge_realdata_correlation_knowledge_application_json"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "benchmark_knowledge_realdata_correlation_realdata_signals_json"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "steps.benchmark_knowledge_readiness.outputs.output_json"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "steps.benchmark_knowledge_application.outputs.output_json"
        in benchmark_knowledge_realdata_script
    )
    assert (
        "steps.benchmark_realdata_signals.outputs.output_json"
        in benchmark_knowledge_realdata_script
    )
    assert "ready_domain_count=" in benchmark_knowledge_realdata_script
    assert "partial_domain_count=" in benchmark_knowledge_realdata_script
    assert "blocked_domain_count=" in benchmark_knowledge_realdata_script
    assert "total_domain_count=" in benchmark_knowledge_realdata_script
    assert "focus_area_count=" in benchmark_knowledge_realdata_script
    assert "focus_areas=" in benchmark_knowledge_realdata_script
    assert "priority_domains=" in benchmark_knowledge_realdata_script
    assert "domain_statuses=" in benchmark_knowledge_realdata_script
    assert "recommendations=" in benchmark_knowledge_realdata_script

    benchmark_knowledge_domain_matrix_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge domain matrix (optional)"
    )
    benchmark_knowledge_domain_matrix_script = benchmark_knowledge_domain_matrix_step["run"]
    assert (
        "scripts/export_benchmark_knowledge_domain_matrix.py"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_ENABLE"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "benchmark_knowledge_domain_matrix_knowledge_readiness_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "benchmark_knowledge_domain_matrix_knowledge_application_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "benchmark_knowledge_domain_matrix_knowledge_realdata_correlation_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "steps.benchmark_knowledge_readiness.outputs.output_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "steps.benchmark_knowledge_application.outputs.output_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert (
        "steps.benchmark_knowledge_realdata_correlation.outputs.output_json"
        in benchmark_knowledge_domain_matrix_script
    )
    assert "ready_domain_count=" in benchmark_knowledge_domain_matrix_script
    assert "partial_domain_count=" in benchmark_knowledge_domain_matrix_script
    assert "blocked_domain_count=" in benchmark_knowledge_domain_matrix_script
    assert "total_domain_count=" in benchmark_knowledge_domain_matrix_script
    assert "focus_area_count=" in benchmark_knowledge_domain_matrix_script
    assert "focus_areas=" in benchmark_knowledge_domain_matrix_script
    assert "priority_domains=" in benchmark_knowledge_domain_matrix_script
    assert "domain_statuses=" in benchmark_knowledge_domain_matrix_script
    assert "recommendations=" in benchmark_knowledge_domain_matrix_script

    benchmark_knowledge_domain_action_plan_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge domain action plan (optional)"
    )
    benchmark_knowledge_domain_action_plan_script = (
        benchmark_knowledge_domain_action_plan_step["run"]
    )
    assert (
        "scripts/export_benchmark_knowledge_domain_action_plan.py"
        in benchmark_knowledge_domain_action_plan_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ENABLE"
        in benchmark_knowledge_domain_action_plan_script
    )
    assert (
        "benchmark_knowledge_domain_action_plan_knowledge_domain_matrix_json"
        in benchmark_knowledge_domain_action_plan_script
    )
    assert "steps.benchmark_knowledge_domain_matrix.outputs.output_json" in (
        benchmark_knowledge_domain_action_plan_script
    )
    assert "ready_domain_count=" in benchmark_knowledge_domain_action_plan_script
    assert "partial_domain_count=" in benchmark_knowledge_domain_action_plan_script
    assert "blocked_domain_count=" in benchmark_knowledge_domain_action_plan_script
    assert "total_domain_count=" in benchmark_knowledge_domain_action_plan_script
    assert "total_action_count=" in benchmark_knowledge_domain_action_plan_script
    assert "high_priority_action_count=" in benchmark_knowledge_domain_action_plan_script
    assert (
        "medium_priority_action_count="
        in benchmark_knowledge_domain_action_plan_script
    )
    assert "priority_domains=" in benchmark_knowledge_domain_action_plan_script
    assert "recommended_first_actions=" in benchmark_knowledge_domain_action_plan_script
    assert "domain_action_counts=" in benchmark_knowledge_domain_action_plan_script
    assert "recommendations=" in benchmark_knowledge_domain_action_plan_script

    benchmark_knowledge_source_action_plan_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge source action plan (optional)"
    )
    benchmark_knowledge_source_action_plan_script = (
        benchmark_knowledge_source_action_plan_step["run"]
    )
    assert (
        "scripts/export_benchmark_knowledge_source_action_plan.py"
        in benchmark_knowledge_source_action_plan_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_ENABLE"
        in benchmark_knowledge_source_action_plan_script
    )
    assert (
        "benchmark_knowledge_source_action_plan_knowledge_source_coverage_json"
        in benchmark_knowledge_source_action_plan_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_KNOWLEDGE_SOURCE_COVERAGE_JSON"
        in benchmark_knowledge_source_action_plan_script
    )
    assert "--benchmark-knowledge-source-coverage" in (
        benchmark_knowledge_source_action_plan_script
    )
    assert "total_action_count=" in benchmark_knowledge_source_action_plan_script
    assert "high_priority_action_count=" in benchmark_knowledge_source_action_plan_script
    assert (
        "medium_priority_action_count="
        in benchmark_knowledge_source_action_plan_script
    )
    assert "expansion_action_count=" in benchmark_knowledge_source_action_plan_script
    assert "priority_domains=" in benchmark_knowledge_source_action_plan_script
    assert (
        "recommended_first_actions="
        in benchmark_knowledge_source_action_plan_script
    )
    assert "source_group_action_counts=" in benchmark_knowledge_source_action_plan_script
    assert "recommendations=" in benchmark_knowledge_source_action_plan_script

    benchmark_knowledge_outcome_correlation_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge outcome correlation (optional)"
    )
    benchmark_knowledge_outcome_correlation_script = (
        benchmark_knowledge_outcome_correlation_step["run"]
    )
    assert (
        "scripts/export_benchmark_knowledge_outcome_correlation.py"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_ENABLE"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert (
        "benchmark_knowledge_outcome_correlation_knowledge_domain_matrix_json"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert (
        "benchmark_knowledge_outcome_correlation_realdata_scorecard_json"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert (
        "steps.benchmark_knowledge_domain_matrix.outputs.output_json"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert (
        "steps.benchmark_realdata_scorecard.outputs.output_json"
        in benchmark_knowledge_outcome_correlation_script
    )
    assert "ready_domain_count=" in benchmark_knowledge_outcome_correlation_script
    assert "partial_domain_count=" in benchmark_knowledge_outcome_correlation_script
    assert "blocked_domain_count=" in benchmark_knowledge_outcome_correlation_script
    assert "total_domain_count=" in benchmark_knowledge_outcome_correlation_script
    assert "focus_area_count=" in benchmark_knowledge_outcome_correlation_script
    assert "focus_areas=" in benchmark_knowledge_outcome_correlation_script
    assert "priority_domains=" in benchmark_knowledge_outcome_correlation_script
    assert "domain_statuses=" in benchmark_knowledge_outcome_correlation_script
    assert "recommendations=" in benchmark_knowledge_outcome_correlation_script

    benchmark_knowledge_outcome_drift_step = _get_step(
        workflow, "evaluate", "Build benchmark knowledge outcome drift (optional)"
    )
    benchmark_knowledge_outcome_drift_script = benchmark_knowledge_outcome_drift_step["run"]
    assert (
        "scripts/export_benchmark_knowledge_outcome_drift.py"
        in benchmark_knowledge_outcome_drift_script
    )
    assert (
        "BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_ENABLE"
        in benchmark_knowledge_outcome_drift_script
    )
    assert (
        "benchmark_knowledge_outcome_drift_current_summary_json"
        in benchmark_knowledge_outcome_drift_script
    )
    assert (
        "benchmark_knowledge_outcome_drift_previous_summary_json"
        in benchmark_knowledge_outcome_drift_script
    )
    assert "steps.benchmark_knowledge_outcome_correlation.outputs.output_json" in (
        benchmark_knowledge_outcome_drift_script
    )
    assert "current_status=" in benchmark_knowledge_outcome_drift_script
    assert "previous_status=" in benchmark_knowledge_outcome_drift_script
    assert "ready_domain_delta=" in benchmark_knowledge_outcome_drift_script
    assert "blocked_domain_delta=" in benchmark_knowledge_outcome_drift_script
    assert "regressions=" in benchmark_knowledge_outcome_drift_script
    assert "improvements=" in benchmark_knowledge_outcome_drift_script
    assert "domain_regressions=" in benchmark_knowledge_outcome_drift_script
    assert "domain_improvements=" in benchmark_knowledge_outcome_drift_script
    assert "resolved_focus_areas=" in benchmark_knowledge_outcome_drift_script
    assert "new_focus_areas=" in benchmark_knowledge_outcome_drift_script
    assert "resolved_priority_domains=" in benchmark_knowledge_outcome_drift_script
    assert "new_priority_domains=" in benchmark_knowledge_outcome_drift_script
    assert "recommendations=" in benchmark_knowledge_outcome_drift_script

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

    benchmark_step = _get_step(workflow, "evaluate", "Generate benchmark scorecard (optional)")
    benchmark_script = benchmark_step["run"]
    assert "scripts/generate_benchmark_scorecard.py" in benchmark_script
    assert "BENCHMARK_SCORECARD_ENABLE" in benchmark_script
    assert "--hybrid-summary" in benchmark_script
    assert "--graph2d-metrics" in benchmark_script
    assert "--history-summary" in benchmark_script
    assert "--assistant-evidence-summary" in benchmark_script
    assert "--review-queue-summary" in benchmark_script
    assert "--feedback-summary" in benchmark_script
    assert "--finetune-summary" in benchmark_script
    assert "--metric-train-summary" in benchmark_script
    assert "--ocr-review-summary" in benchmark_script
    assert "--qdrant-readiness-summary" in benchmark_script
    assert "--engineering-signals-summary" in benchmark_script
    assert "--knowledge-readiness-summary" in benchmark_script
    assert "--benchmark-operator-adoption-summary" in benchmark_script
    assert "overall_status=" in benchmark_script
    assert "assistant_status=" in benchmark_script
    assert "review_queue_status=" in benchmark_script
    assert "feedback_flywheel_status=" in benchmark_script
    assert "ocr_status=" in benchmark_script
    assert "qdrant_status=" in benchmark_script
    assert "knowledge_status=" in benchmark_script
    assert "knowledge_total_reference_items=" in benchmark_script
    assert "knowledge_focus_area_count=" in benchmark_script
    assert "knowledge_focus_areas=" in benchmark_script
    assert "engineering_status=" in benchmark_script
    assert "engineering_coverage_ratio=" in benchmark_script
    assert "engineering_top_standard_types=" in benchmark_script
    assert "operator_adoption_status=" in benchmark_script
    assert "operator_adoption_mode=" in benchmark_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_script

    feedback_flywheel_step = _get_step(
        workflow, "evaluate", "Build feedback flywheel benchmark artifact (optional)"
    )
    feedback_flywheel_script = feedback_flywheel_step["run"]
    assert "scripts/export_feedback_flywheel_benchmark.py" in feedback_flywheel_script
    assert "--feedback-summary" in feedback_flywheel_script
    assert "--finetune-summary" in feedback_flywheel_script
    assert "--metric-train-summary" in feedback_flywheel_script
    assert "INPUT_COUNT=0" in feedback_flywheel_script
    assert "feedback_total=" in feedback_flywheel_script
    assert "metric_triplet_count=" in feedback_flywheel_script

    benchmark_operational_step = _get_step(
        workflow, "evaluate", "Build benchmark operational summary (optional)"
    )
    benchmark_operational_script = benchmark_operational_step["run"]
    assert "scripts/export_benchmark_operational_summary.py" in benchmark_operational_script
    assert "BENCHMARK_OPERATIONAL_SUMMARY_ENABLE" in benchmark_operational_script
    assert "--benchmark-scorecard" in benchmark_operational_script
    assert "--feedback-flywheel" in benchmark_operational_script
    assert "--assistant-evidence" in benchmark_operational_script
    assert "--review-queue" in benchmark_operational_script
    assert "--ocr-review" in benchmark_operational_script
    assert "--benchmark-operator-adoption" in benchmark_operational_script
    assert "INPUT_COUNT=0" in benchmark_operational_script
    assert "feedback_status=" in benchmark_operational_script
    assert "assistant_status=" in benchmark_operational_script
    assert "review_queue_status=" in benchmark_operational_script
    assert "ocr_status=" in benchmark_operational_script
    assert "operator_adoption_status=" in benchmark_operational_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_operational_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_operational_script
    assert "blockers=" in benchmark_operational_script
    assert "recommendations=" in benchmark_operational_script

    benchmark_bundle_step = _get_step(
        workflow, "evaluate", "Build benchmark artifact bundle (optional)"
    )
    benchmark_bundle_script = benchmark_bundle_step["run"]
    assert "scripts/export_benchmark_artifact_bundle.py" in benchmark_bundle_script
    assert "BENCHMARK_ARTIFACT_BUNDLE_ENABLE" in benchmark_bundle_script
    assert "--benchmark-scorecard" in benchmark_bundle_script
    assert "--benchmark-operational-summary" in benchmark_bundle_script
    assert "--feedback-flywheel" in benchmark_bundle_script
    assert "--assistant-evidence" in benchmark_bundle_script
    assert "--review-queue" in benchmark_bundle_script
    assert "--ocr-review" in benchmark_bundle_script
    assert "--benchmark-companion-summary" in benchmark_bundle_script
    assert "--benchmark-release-decision" in benchmark_bundle_script
    assert "--benchmark-engineering-signals" in benchmark_bundle_script
    assert "--benchmark-realdata-signals" in benchmark_bundle_script
    assert "--benchmark-realdata-scorecard" in benchmark_bundle_script
    assert "--benchmark-knowledge-readiness" in benchmark_bundle_script
    assert "--benchmark-knowledge-drift" in benchmark_bundle_script
    assert "--benchmark-knowledge-application" in benchmark_bundle_script
    assert "--benchmark-knowledge-domain-action-plan" in benchmark_bundle_script
    assert "--benchmark-knowledge-source-action-plan" in benchmark_bundle_script
    assert "--benchmark-knowledge-outcome-correlation" in benchmark_bundle_script
    assert "--benchmark-knowledge-outcome-drift" in benchmark_bundle_script
    assert "--benchmark-competitive-surpass-index" in benchmark_bundle_script
    assert "INPUT_COUNT=0" in benchmark_bundle_script
    assert "available_artifact_count=" in benchmark_bundle_script
    assert "feedback_status=" in benchmark_bundle_script
    assert "assistant_status=" in benchmark_bundle_script
    assert "review_queue_status=" in benchmark_bundle_script
    assert "ocr_status=" in benchmark_bundle_script
    assert "knowledge_status=" in benchmark_bundle_script
    assert "knowledge_drift_status=" in benchmark_bundle_script
    assert "knowledge_drift_summary=" in benchmark_bundle_script
    assert "knowledge_drift_recommendations=" in benchmark_bundle_script
    assert "knowledge_drift_component_changes=" in benchmark_bundle_script
    assert "knowledge_drift_domain_regressions=" in benchmark_bundle_script
    assert "knowledge_drift_domain_improvements=" in benchmark_bundle_script
    assert "knowledge_drift_resolved_priority_domains=" in benchmark_bundle_script
    assert "knowledge_drift_new_priority_domains=" in benchmark_bundle_script
    assert "engineering_status=" in benchmark_bundle_script
    assert "realdata_status=" in benchmark_bundle_script
    assert "realdata_scorecard_status=" in benchmark_bundle_script
    assert "realdata_scorecard_recommendations=" in benchmark_bundle_script
    assert "realdata_recommendations=" in benchmark_bundle_script
    assert "knowledge_application_status=" in benchmark_bundle_script
    assert "knowledge_application_focus_areas=" in benchmark_bundle_script
    assert "knowledge_application_priority_domains=" in benchmark_bundle_script
    assert "knowledge_application_domain_statuses=" in benchmark_bundle_script
    assert "knowledge_application_recommendations=" in benchmark_bundle_script
    assert "knowledge_domain_action_plan_status=" in benchmark_bundle_script
    assert "knowledge_domain_action_plan_actions=" in benchmark_bundle_script
    assert "knowledge_domain_action_plan_priority_domains=" in benchmark_bundle_script
    assert "knowledge_domain_action_plan_recommendations=" in benchmark_bundle_script
    assert "knowledge_source_action_plan_status=" in benchmark_bundle_script
    assert "knowledge_source_action_plan_priority_domains=" in benchmark_bundle_script
    assert "knowledge_source_action_plan_recommended_first_actions=" in (
        benchmark_bundle_script
    )
    assert "knowledge_source_action_plan_source_group_action_counts=" in (
        benchmark_bundle_script
    )
    assert "knowledge_source_action_plan_recommendations=" in benchmark_bundle_script
    assert "knowledge_outcome_correlation_status=" in benchmark_bundle_script
    assert "knowledge_outcome_correlation_focus_areas=" in benchmark_bundle_script
    assert "knowledge_outcome_correlation_priority_domains=" in benchmark_bundle_script
    assert "knowledge_outcome_correlation_domain_statuses=" in benchmark_bundle_script
    assert "knowledge_outcome_correlation_recommendations=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_status=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_summary=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_domain_regressions=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_domain_improvements=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_resolved_priority_domains=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_new_priority_domains=" in benchmark_bundle_script
    assert "knowledge_outcome_drift_recommendations=" in benchmark_bundle_script
    assert "competitive_surpass_index_status=" in benchmark_bundle_script
    assert "competitive_surpass_primary_gaps=" in benchmark_bundle_script
    assert "competitive_surpass_recommendations=" in benchmark_bundle_script
    assert "operator_adoption_knowledge_drift_status=" in benchmark_bundle_script
    assert "operator_adoption_knowledge_drift_summary=" in benchmark_bundle_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_bundle_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_bundle_script
    assert "scorecard_operator_adoption_status=" in benchmark_bundle_script
    assert "scorecard_operator_adoption_mode=" in benchmark_bundle_script
    assert "scorecard_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_bundle_script
    )
    assert "operational_operator_adoption_status=" in benchmark_bundle_script
    assert "operational_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_bundle_script
    )
    assert "blockers=" in benchmark_bundle_script
    assert "recommendations=" in benchmark_bundle_script

    benchmark_companion_step = _get_step(
        workflow, "evaluate", "Build benchmark companion summary (optional)"
    )
    benchmark_companion_script = benchmark_companion_step["run"]
    assert "scripts/export_benchmark_companion_summary.py" in benchmark_companion_script
    assert "BENCHMARK_COMPANION_SUMMARY_ENABLE" in benchmark_companion_script
    assert "--benchmark-scorecard" in benchmark_companion_script
    assert "--benchmark-operational-summary" in benchmark_companion_script
    assert "--benchmark-artifact-bundle" in benchmark_companion_script
    assert "--benchmark-engineering-signals" in benchmark_companion_script
    assert "--benchmark-realdata-signals" in benchmark_companion_script
    assert "--benchmark-realdata-scorecard" in benchmark_companion_script
    assert "--benchmark-knowledge-readiness" in benchmark_companion_script
    assert "--benchmark-knowledge-drift" in benchmark_companion_script
    assert "--benchmark-knowledge-application" in benchmark_companion_script
    assert "--benchmark-knowledge-domain-action-plan" in benchmark_companion_script
    assert "--benchmark-knowledge-source-action-plan" in benchmark_companion_script
    assert "--benchmark-knowledge-outcome-correlation" in benchmark_companion_script
    assert "--benchmark-knowledge-outcome-drift" in benchmark_companion_script
    assert "--benchmark-competitive-surpass-index" in benchmark_companion_script
    assert "review_surface=" in benchmark_companion_script
    assert "primary_gap=" in benchmark_companion_script
    assert "recommended_actions=" in benchmark_companion_script
    assert "qdrant_status=" in benchmark_companion_script
    assert "knowledge_status=" in benchmark_companion_script
    assert "knowledge_drift_status=" in benchmark_companion_script
    assert "knowledge_drift_summary=" in benchmark_companion_script
    assert "knowledge_drift_recommendations=" in benchmark_companion_script
    assert "knowledge_drift_component_changes=" in benchmark_companion_script
    assert "knowledge_drift_domain_regressions=" in benchmark_companion_script
    assert "knowledge_drift_domain_improvements=" in benchmark_companion_script
    assert "knowledge_drift_resolved_priority_domains=" in benchmark_companion_script
    assert "knowledge_drift_new_priority_domains=" in benchmark_companion_script
    assert "engineering_status=" in benchmark_companion_script
    assert "realdata_status=" in benchmark_companion_script
    assert "realdata_scorecard_status=" in benchmark_companion_script
    assert "realdata_scorecard_recommendations=" in benchmark_companion_script
    assert "realdata_recommendations=" in benchmark_companion_script
    assert "knowledge_application_status=" in benchmark_companion_script
    assert "knowledge_application_focus_areas=" in benchmark_companion_script
    assert "knowledge_application_priority_domains=" in benchmark_companion_script
    assert "knowledge_application_domain_statuses=" in benchmark_companion_script
    assert "knowledge_application_recommendations=" in benchmark_companion_script
    assert "knowledge_domain_action_plan_status=" in benchmark_companion_script
    assert "knowledge_domain_action_plan_actions=" in benchmark_companion_script
    assert "knowledge_domain_action_plan_priority_domains=" in benchmark_companion_script
    assert "knowledge_domain_action_plan_recommendations=" in benchmark_companion_script
    assert "knowledge_source_action_plan_status=" in benchmark_companion_script
    assert "knowledge_source_action_plan_priority_domains=" in benchmark_companion_script
    assert "knowledge_source_action_plan_recommended_first_actions=" in (
        benchmark_companion_script
    )
    assert "knowledge_source_action_plan_source_group_action_counts=" in (
        benchmark_companion_script
    )
    assert "knowledge_source_action_plan_recommendations=" in benchmark_companion_script
    assert "knowledge_outcome_correlation_status=" in benchmark_companion_script
    assert "knowledge_outcome_correlation_focus_areas=" in benchmark_companion_script
    assert "knowledge_outcome_correlation_priority_domains=" in benchmark_companion_script
    assert "knowledge_outcome_correlation_domain_statuses=" in benchmark_companion_script
    assert "knowledge_outcome_correlation_recommendations=" in benchmark_companion_script
    assert "knowledge_outcome_drift_status=" in benchmark_companion_script
    assert "knowledge_outcome_drift_summary=" in benchmark_companion_script
    assert "knowledge_outcome_drift_domain_regressions=" in benchmark_companion_script
    assert "knowledge_outcome_drift_domain_improvements=" in benchmark_companion_script
    assert "knowledge_outcome_drift_resolved_priority_domains=" in benchmark_companion_script
    assert "knowledge_outcome_drift_new_priority_domains=" in benchmark_companion_script
    assert "knowledge_outcome_drift_recommendations=" in benchmark_companion_script
    assert "competitive_surpass_index_status=" in benchmark_companion_script
    assert "competitive_surpass_primary_gaps=" in benchmark_companion_script
    assert "competitive_surpass_recommendations=" in benchmark_companion_script
    assert "operator_adoption_knowledge_drift_status=" in benchmark_companion_script
    assert "operator_adoption_knowledge_drift_summary=" in benchmark_companion_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_companion_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_companion_script
    assert "scorecard_operator_adoption_status=" in benchmark_companion_script
    assert "scorecard_operator_adoption_mode=" in benchmark_companion_script
    assert "scorecard_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_companion_script
    )
    assert "operational_operator_adoption_status=" in benchmark_companion_script
    assert "operational_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_companion_script
    )

    benchmark_release_step = _get_step(
        workflow, "evaluate", "Build benchmark release decision (optional)"
    )
    benchmark_release_script = benchmark_release_step["run"]
    assert "scripts/export_benchmark_release_decision.py" in benchmark_release_script
    assert "BENCHMARK_RELEASE_DECISION_ENABLE" in benchmark_release_script
    assert "--benchmark-scorecard" in benchmark_release_script
    assert "--benchmark-operational-summary" in benchmark_release_script
    assert "--benchmark-artifact-bundle" in benchmark_release_script
    assert "--benchmark-companion-summary" in benchmark_release_script
    assert "--benchmark-engineering-signals" in benchmark_release_script
    assert "--benchmark-realdata-signals" in benchmark_release_script
    assert "--benchmark-realdata-scorecard" in benchmark_release_script
    assert "--benchmark-operator-adoption" in benchmark_release_script
    assert "--benchmark-knowledge-readiness" in benchmark_release_script
    assert "--benchmark-knowledge-drift" in benchmark_release_script
    assert "--benchmark-knowledge-application" in benchmark_release_script
    assert "--benchmark-knowledge-domain-action-plan" in benchmark_release_script
    assert "--benchmark-knowledge-source-action-plan" in benchmark_release_script
    assert "--benchmark-knowledge-outcome-correlation" in benchmark_release_script
    assert "--benchmark-knowledge-outcome-drift" in benchmark_release_script
    assert "--benchmark-competitive-surpass-index" in benchmark_release_script
    assert "release_status=" in benchmark_release_script
    assert "automation_ready=" in benchmark_release_script
    assert "primary_signal_source=" in benchmark_release_script
    assert "blocking_signals=" in benchmark_release_script
    assert "review_signals=" in benchmark_release_script
    assert "qdrant_status=" in benchmark_release_script
    assert "knowledge_status=" in benchmark_release_script
    assert "knowledge_drift_status=" in benchmark_release_script
    assert "knowledge_drift_summary=" in benchmark_release_script
    assert "knowledge_drift_domain_regressions=" in benchmark_release_script
    assert "knowledge_drift_domain_improvements=" in benchmark_release_script
    assert "knowledge_drift_resolved_priority_domains=" in benchmark_release_script
    assert "knowledge_drift_new_priority_domains=" in benchmark_release_script
    assert "engineering_status=" in benchmark_release_script
    assert "realdata_status=" in benchmark_release_script
    assert "realdata_scorecard_status=" in benchmark_release_script
    assert "realdata_scorecard_recommendations=" in benchmark_release_script
    assert "realdata_recommendations=" in benchmark_release_script
    assert "knowledge_application_status=" in benchmark_release_script
    assert "knowledge_application_focus_areas=" in benchmark_release_script
    assert "knowledge_application_priority_domains=" in benchmark_release_script
    assert "knowledge_application_domain_statuses=" in benchmark_release_script
    assert "knowledge_application_recommendations=" in benchmark_release_script
    assert "knowledge_domain_action_plan_status=" in benchmark_release_script
    assert "knowledge_domain_action_plan_actions=" in benchmark_release_script
    assert "knowledge_domain_action_plan_priority_domains=" in benchmark_release_script
    assert "knowledge_domain_action_plan_recommendations=" in benchmark_release_script
    assert "knowledge_source_action_plan_status=" in benchmark_release_script
    assert "knowledge_source_action_plan_priority_domains=" in benchmark_release_script
    assert "knowledge_source_action_plan_recommended_first_actions=" in (
        benchmark_release_script
    )
    assert "knowledge_source_action_plan_source_group_action_counts=" in (
        benchmark_release_script
    )
    assert "knowledge_source_action_plan_recommendations=" in benchmark_release_script
    assert "knowledge_outcome_correlation_status=" in benchmark_release_script
    assert "knowledge_outcome_correlation_focus_areas=" in benchmark_release_script
    assert "knowledge_outcome_correlation_priority_domains=" in benchmark_release_script
    assert "knowledge_outcome_correlation_domain_statuses=" in benchmark_release_script
    assert "knowledge_outcome_correlation_recommendations=" in benchmark_release_script
    assert "knowledge_outcome_drift_status=" in benchmark_release_script
    assert "knowledge_outcome_drift_summary=" in benchmark_release_script
    assert "knowledge_outcome_drift_domain_regressions=" in benchmark_release_script
    assert "knowledge_outcome_drift_domain_improvements=" in benchmark_release_script
    assert "knowledge_outcome_drift_resolved_priority_domains=" in benchmark_release_script
    assert "knowledge_outcome_drift_new_priority_domains=" in benchmark_release_script
    assert "knowledge_outcome_drift_recommendations=" in benchmark_release_script
    assert "competitive_surpass_index_status=" in benchmark_release_script
    assert "competitive_surpass_primary_gaps=" in benchmark_release_script
    assert "competitive_surpass_recommendations=" in benchmark_release_script
    assert "operator_adoption_status=" in benchmark_release_script
    assert "operator_adoption_knowledge_drift_status=" in benchmark_release_script
    assert "operator_adoption_knowledge_drift_summary=" in benchmark_release_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_release_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_release_script
    assert "scorecard_operator_adoption_status=" in benchmark_release_script
    assert "scorecard_operator_adoption_mode=" in benchmark_release_script
    assert "scorecard_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_release_script
    )
    assert "operational_operator_adoption_status=" in benchmark_release_script
    assert "operational_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_release_script
    )

    benchmark_runbook_step = _get_step(
        workflow, "evaluate", "Build benchmark release runbook (optional)"
    )
    benchmark_runbook_script = benchmark_runbook_step["run"]
    assert "scripts/export_benchmark_release_runbook.py" in benchmark_runbook_script
    assert "BENCHMARK_RELEASE_RUNBOOK_ENABLE" in benchmark_runbook_script
    assert "--benchmark-release-decision" in benchmark_runbook_script
    assert "--benchmark-scorecard" in benchmark_runbook_script
    assert "--benchmark-operational-summary" in benchmark_runbook_script
    assert "--benchmark-companion-summary" in benchmark_runbook_script
    assert "--benchmark-artifact-bundle" in benchmark_runbook_script
    assert "--benchmark-engineering-signals" in benchmark_runbook_script
    assert "--benchmark-realdata-signals" in benchmark_runbook_script
    assert "--benchmark-realdata-scorecard" in benchmark_runbook_script
    assert "--benchmark-operator-adoption" in benchmark_runbook_script
    assert "--benchmark-knowledge-readiness" in benchmark_runbook_script
    assert "--benchmark-knowledge-drift" in benchmark_runbook_script
    assert "--benchmark-knowledge-application" in benchmark_runbook_script
    assert "--benchmark-knowledge-domain-action-plan" in benchmark_runbook_script
    assert "--benchmark-knowledge-source-action-plan" in benchmark_runbook_script
    assert "--benchmark-knowledge-outcome-correlation" in benchmark_runbook_script
    assert "--benchmark-knowledge-outcome-drift" in benchmark_runbook_script
    assert "--benchmark-competitive-surpass-index" in benchmark_runbook_script
    assert "ready_to_freeze_baseline=" in benchmark_runbook_script
    assert "next_action=" in benchmark_runbook_script
    assert "missing_artifacts=" in benchmark_runbook_script
    assert "blocking_signals=" in benchmark_runbook_script
    assert "review_signals=" in benchmark_runbook_script
    assert "knowledge_status=" in benchmark_runbook_script
    assert "knowledge_drift_status=" in benchmark_runbook_script
    assert "knowledge_drift_summary=" in benchmark_runbook_script
    assert "knowledge_drift_domain_regressions=" in benchmark_runbook_script
    assert "knowledge_drift_domain_improvements=" in benchmark_runbook_script
    assert "knowledge_drift_resolved_priority_domains=" in benchmark_runbook_script
    assert "knowledge_drift_new_priority_domains=" in benchmark_runbook_script
    assert "engineering_status=" in benchmark_runbook_script
    assert "realdata_status=" in benchmark_runbook_script
    assert "realdata_scorecard_status=" in benchmark_runbook_script
    assert "realdata_scorecard_recommendations=" in benchmark_runbook_script
    assert "realdata_recommendations=" in benchmark_runbook_script
    assert "knowledge_application_status=" in benchmark_runbook_script
    assert "knowledge_application_focus_areas=" in benchmark_runbook_script
    assert "knowledge_application_priority_domains=" in benchmark_runbook_script
    assert "knowledge_application_domain_statuses=" in benchmark_runbook_script
    assert "knowledge_application_recommendations=" in benchmark_runbook_script
    assert "knowledge_domain_action_plan_status=" in benchmark_runbook_script
    assert "knowledge_domain_action_plan_actions=" in benchmark_runbook_script
    assert "knowledge_domain_action_plan_priority_domains=" in benchmark_runbook_script
    assert "knowledge_domain_action_plan_recommendations=" in benchmark_runbook_script
    assert "knowledge_source_action_plan_status=" in benchmark_runbook_script
    assert "knowledge_source_action_plan_priority_domains=" in benchmark_runbook_script
    assert "knowledge_source_action_plan_recommended_first_actions=" in (
        benchmark_runbook_script
    )
    assert "knowledge_source_action_plan_source_group_action_counts=" in (
        benchmark_runbook_script
    )
    assert "knowledge_source_action_plan_recommendations=" in benchmark_runbook_script
    assert "knowledge_outcome_correlation_status=" in benchmark_runbook_script
    assert "knowledge_outcome_correlation_focus_areas=" in benchmark_runbook_script
    assert "knowledge_outcome_correlation_priority_domains=" in benchmark_runbook_script
    assert "knowledge_outcome_correlation_domain_statuses=" in benchmark_runbook_script
    assert "knowledge_outcome_correlation_recommendations=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_status=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_summary=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_domain_regressions=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_domain_improvements=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_resolved_priority_domains=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_new_priority_domains=" in benchmark_runbook_script
    assert "knowledge_outcome_drift_recommendations=" in benchmark_runbook_script
    assert "competitive_surpass_index_status=" in benchmark_runbook_script
    assert "competitive_surpass_primary_gaps=" in benchmark_runbook_script
    assert "competitive_surpass_recommendations=" in benchmark_runbook_script
    assert "operator_adoption_status=" in benchmark_runbook_script
    assert "operator_adoption_knowledge_drift_status=" in benchmark_runbook_script
    assert "operator_adoption_knowledge_drift_summary=" in benchmark_runbook_script
    assert "operator_adoption_knowledge_outcome_drift_status=" in benchmark_runbook_script
    assert "operator_adoption_knowledge_outcome_drift_summary=" in benchmark_runbook_script
    assert "scorecard_operator_adoption_status=" in benchmark_runbook_script
    assert "scorecard_operator_adoption_mode=" in benchmark_runbook_script
    assert "scorecard_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_runbook_script
    )
    assert "operational_operator_adoption_status=" in benchmark_runbook_script
    assert "operational_operator_adoption_knowledge_outcome_drift_status=" in (
        benchmark_runbook_script
    )
    assert "Benchmark Release Decision Competitive Surpass" in workflow_text
    assert "Benchmark Release Runbook Competitive Surpass" in workflow_text

    benchmark_operator_adoption_step = _get_step(
        workflow, "evaluate", "Build benchmark operator adoption (optional)"
    )
    benchmark_operator_adoption_script = benchmark_operator_adoption_step["run"]
    assert (
        "scripts/export_benchmark_operator_adoption.py"
        in benchmark_operator_adoption_script
    )
    assert "BENCHMARK_OPERATOR_ADOPTION_ENABLE" in benchmark_operator_adoption_script
    assert "--benchmark-release-decision" in benchmark_operator_adoption_script
    assert "--benchmark-release-runbook" in benchmark_operator_adoption_script
    assert "--review-queue" in benchmark_operator_adoption_script
    assert "--feedback-flywheel" in benchmark_operator_adoption_script
    assert "--benchmark-knowledge-drift" in benchmark_operator_adoption_script
    assert "--benchmark-knowledge-outcome-drift" in benchmark_operator_adoption_script
    assert "INPUT_COUNT=0" in benchmark_operator_adoption_script
    assert "adoption_readiness=" in benchmark_operator_adoption_script
    assert "operator_mode=" in benchmark_operator_adoption_script
    assert "next_action=" in benchmark_operator_adoption_script
    assert "automation_ready=" in benchmark_operator_adoption_script
    assert "freeze_ready=" in benchmark_operator_adoption_script
    assert "release_status=" in benchmark_operator_adoption_script
    assert "runbook_status=" in benchmark_operator_adoption_script
    assert "review_queue_status=" in benchmark_operator_adoption_script
    assert "feedback_status=" in benchmark_operator_adoption_script
    assert "knowledge_drift_status=" in benchmark_operator_adoption_script
    assert "knowledge_drift_summary=" in benchmark_operator_adoption_script
    assert "knowledge_outcome_drift_status=" in benchmark_operator_adoption_script
    assert "knowledge_outcome_drift_summary=" in benchmark_operator_adoption_script
    assert "blocking_signals=" in benchmark_operator_adoption_script
    assert "recommended_actions=" in benchmark_operator_adoption_script

    benchmark_competitive_surpass_step = _get_step(
        workflow, "evaluate", "Build benchmark competitive surpass index (optional)"
    )
    benchmark_competitive_surpass_script = benchmark_competitive_surpass_step["run"]
    assert (
        "scripts/export_benchmark_competitive_surpass_index.py"
        in benchmark_competitive_surpass_script
    )
    assert (
        "BENCHMARK_COMPETITIVE_SURPASS_INDEX_ENABLE"
        in benchmark_competitive_surpass_script
    )
    assert "--benchmark-engineering-signals" in benchmark_competitive_surpass_script
    assert "--benchmark-knowledge-readiness" in benchmark_competitive_surpass_script
    assert "--benchmark-knowledge-application" in benchmark_competitive_surpass_script
    assert (
        "--benchmark-knowledge-realdata-correlation"
        in benchmark_competitive_surpass_script
    )
    assert "--benchmark-knowledge-domain-matrix" in benchmark_competitive_surpass_script
    assert (
        "--benchmark-knowledge-outcome-correlation"
        in benchmark_competitive_surpass_script
    )
    assert "--benchmark-knowledge-outcome-drift" in benchmark_competitive_surpass_script
    assert "--benchmark-realdata-signals" in benchmark_competitive_surpass_script
    assert "--benchmark-realdata-scorecard" in benchmark_competitive_surpass_script
    assert "--benchmark-operator-adoption" in benchmark_competitive_surpass_script
    assert "INPUT_COUNT=0" in benchmark_competitive_surpass_script
    assert "status=" in benchmark_competitive_surpass_script
    assert "score=" in benchmark_competitive_surpass_script
    assert "ready_pillars=" in benchmark_competitive_surpass_script
    assert "partial_pillars=" in benchmark_competitive_surpass_script
    assert "blocked_pillars=" in benchmark_competitive_surpass_script
    assert "primary_gaps=" in benchmark_competitive_surpass_script
    assert "recommendations=" in benchmark_competitive_surpass_script

    assistant_step = _get_step(
        workflow, "evaluate", "Build assistant evidence report (optional)"
    )
    assistant_script = assistant_step["run"]
    assert "scripts/export_assistant_evidence_report.py" in assistant_script
    assert "ASSISTANT_EVIDENCE_REPORT_ENABLE" in assistant_script
    assert "assistant_evidence_report_input" in assistant_script
    assert "average_evidence_count=" in assistant_script
    assert "top_evidence_types=" in assistant_script
    assert "top_missing_fields=" in assistant_script

    review_queue_step = _get_step(
        workflow, "evaluate", "Build active-learning review queue report (optional)"
    )
    review_queue_script = review_queue_step["run"]
    assert "scripts/export_active_learning_review_queue_report.py" in review_queue_script
    assert "ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLE" in review_queue_script
    assert "active_learning_review_queue_report_input" in review_queue_script
    assert "--top-k" in review_queue_script
    assert "operational_status=" in review_queue_script
    assert "top_feedback_priorities=" in review_queue_script
    assert "top_decision_sources=" in review_queue_script
    assert "top_review_reasons=" in review_queue_script

    review_queue_summary_flag = (
        '--review-queue-summary '
        '"${{ steps.active_learning_review_queue_report.outputs.output_json || \'\' }}"'
    )
    assert review_queue_summary_flag in benchmark_script
    engineering_summary_flag = (
        '--engineering-signals-summary '
        '"${{ steps.benchmark_engineering_signals.outputs.output_json || \'\' }}"'
    )
    assert engineering_summary_flag in benchmark_script
    knowledge_summary_flag = (
        '--knowledge-readiness-summary '
        '"${{ steps.benchmark_knowledge_readiness.outputs.output_json || \'\' }}"'
    )
    assert knowledge_summary_flag in benchmark_script


def test_workflow_uploads_new_graph2d_artifacts_and_summary_lines() -> None:
    workflow = _load_workflow()

    upload_review = _get_step(workflow, "evaluate", "Upload Graph2D review pack")
    assert upload_review["if"] == "steps.graph2d_review_pack.outputs.enabled == 'true'"

    upload_sweep = _get_step(workflow, "evaluate", "Upload Graph2D train sweep")
    assert upload_sweep["if"] == "steps.graph2d_train_sweep.outputs.enabled == 'true'"

    upload_scorecard = _get_step(workflow, "evaluate", "Upload benchmark scorecard")
    assert upload_scorecard["if"] == "steps.benchmark_scorecard.outputs.enabled == 'true'"
    upload_engineering = _get_step(workflow, "evaluate", "Upload benchmark engineering signals")
    assert (
        upload_engineering["if"]
        == "steps.benchmark_engineering_signals.outputs.enabled == 'true'"
    )
    upload_realdata = _get_step(workflow, "evaluate", "Upload benchmark realdata signals")
    assert (
        upload_realdata["if"]
        == "steps.benchmark_realdata_signals.outputs.enabled == 'true'"
    )
    upload_realdata_scorecard = _get_step(
        workflow, "evaluate", "Upload benchmark realdata scorecard"
    )
    assert (
        upload_realdata_scorecard["if"]
        == "steps.benchmark_realdata_scorecard.outputs.enabled == 'true'"
    )
    upload_knowledge = _get_step(workflow, "evaluate", "Upload benchmark knowledge readiness")
    assert (
        upload_knowledge["if"]
        == "steps.benchmark_knowledge_readiness.outputs.enabled == 'true'"
    )
    upload_knowledge_drift = _get_step(workflow, "evaluate", "Upload benchmark knowledge drift")
    assert (
        upload_knowledge_drift["if"]
        == "steps.benchmark_knowledge_drift.outputs.enabled == 'true'"
    )
    upload_knowledge_application = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge application"
    )
    assert (
        upload_knowledge_application["if"]
        == "steps.benchmark_knowledge_application.outputs.enabled == 'true'"
    )
    upload_knowledge_realdata = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge realdata correlation"
    )
    assert (
        upload_knowledge_realdata["if"]
        == "steps.benchmark_knowledge_realdata_correlation.outputs.enabled == 'true'"
    )
    upload_knowledge_domain_matrix = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge domain matrix"
    )
    assert (
        upload_knowledge_domain_matrix["if"]
        == "steps.benchmark_knowledge_domain_matrix.outputs.enabled == 'true'"
    )
    upload_knowledge_domain_action_plan = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge domain action plan"
    )
    assert (
        upload_knowledge_domain_action_plan["if"]
        == "steps.benchmark_knowledge_domain_action_plan.outputs.enabled == 'true'"
    )
    upload_knowledge_source_action_plan = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge source action plan"
    )
    assert (
        upload_knowledge_source_action_plan["if"]
        == "steps.benchmark_knowledge_source_action_plan.outputs.enabled == 'true'"
    )
    upload_knowledge_outcome_correlation = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge outcome correlation"
    )
    assert (
        upload_knowledge_outcome_correlation["if"]
        == "steps.benchmark_knowledge_outcome_correlation.outputs.enabled == 'true'"
    )
    upload_knowledge_outcome_drift = _get_step(
        workflow, "evaluate", "Upload benchmark knowledge outcome drift"
    )
    assert (
        upload_knowledge_outcome_drift["if"]
        == "steps.benchmark_knowledge_outcome_drift.outputs.enabled == 'true'"
    )
    upload_feedback_flywheel = _get_step(
        workflow, "evaluate", "Upload feedback flywheel benchmark artifact"
    )
    assert (
        upload_feedback_flywheel["if"]
        == "steps.feedback_flywheel_benchmark.outputs.enabled == 'true'"
    )
    upload_benchmark_operational = _get_step(
        workflow, "evaluate", "Upload benchmark operational summary"
    )
    assert (
        upload_benchmark_operational["if"]
        == "steps.benchmark_operational_summary.outputs.enabled == 'true'"
    )
    upload_benchmark_competitive_surpass = _get_step(
        workflow, "evaluate", "Upload benchmark competitive surpass index"
    )
    assert (
        upload_benchmark_competitive_surpass["if"]
        == "steps.benchmark_competitive_surpass_index.outputs.enabled == 'true'"
    )
    upload_benchmark_bundle = _get_step(
        workflow, "evaluate", "Upload benchmark artifact bundle"
    )
    assert (
        upload_benchmark_bundle["if"]
        == "steps.benchmark_artifact_bundle.outputs.enabled == 'true'"
    )
    upload_benchmark_companion = _get_step(
        workflow, "evaluate", "Upload benchmark companion summary"
    )
    assert (
        upload_benchmark_companion["if"]
        == "steps.benchmark_companion_summary.outputs.enabled == 'true'"
    )
    upload_benchmark_release = _get_step(
        workflow, "evaluate", "Upload benchmark release decision"
    )
    assert (
        upload_benchmark_release["if"]
        == "steps.benchmark_release_decision.outputs.enabled == 'true'"
    )
    upload_benchmark_runbook = _get_step(
        workflow, "evaluate", "Upload benchmark release runbook"
    )
    assert (
        upload_benchmark_runbook["if"]
        == "steps.benchmark_release_runbook.outputs.enabled == 'true'"
    )
    upload_benchmark_operator_adoption = _get_step(
        workflow, "evaluate", "Upload benchmark operator adoption"
    )
    assert (
        upload_benchmark_operator_adoption["if"]
        == "steps.benchmark_operator_adoption.outputs.enabled == 'true'"
    )
    upload_assistant = _get_step(workflow, "evaluate", "Upload assistant evidence report")
    assert (
        upload_assistant["if"]
        == "steps.assistant_evidence_report.outputs.enabled == 'true'"
    )
    upload_review_queue = _get_step(
        workflow, "evaluate", "Upload active-learning review queue report"
    )
    assert (
        upload_review_queue["if"]
        == "steps.active_learning_review_queue_report.outputs.enabled == 'true'"
    )
    upload_ocr_review = _get_step(workflow, "evaluate", "Upload OCR review pack")
    assert upload_ocr_review["if"] == "steps.ocr_review_pack.outputs.enabled == 'true'"

    summary_step = _get_step(workflow, "evaluate", "Create job summary")
    summary_script = summary_step["run"]
    assert "Graph2D review input" in summary_script
    assert "Graph2D review candidates" in summary_script
    assert "Graph2D review gate status" in summary_script
    assert "Graph2D review gate headline" in summary_script
    assert "Graph2D review top reasons" in summary_script
    assert "Graph2D review priorities" in summary_script
    assert "Graph2D review confidence bands" in summary_script
    assert "Graph2D review coarse labels" in summary_script
    assert "Graph2D review fine labels" in summary_script
    assert "Graph2D review rejection reasons" in summary_script
    assert "Graph2D review knowledge_conflicts" in summary_script
    assert "Graph2D review knowledge conflict details" in summary_script
    assert "Graph2D review knowledge rows" in summary_script
    assert "Graph2D review standards rows" in summary_script
    assert "Graph2D review knowledge categories" in summary_script
    assert "Graph2D review standard candidates" in summary_script
    assert "Graph2D review knowledge hints" in summary_script
    assert "Graph2D review top sources" in summary_script
    assert "Graph2D review shadow sources" in summary_script
    assert "Graph2D review example explanations" in summary_script
    assert "Graph2D review gate strict_mode" in summary_script
    assert "Graph2D train sweep total_runs" in summary_script
    assert "Graph2D train sweep best run script" in summary_script
    assert "Benchmark scorecard overall" in summary_script
    assert "Benchmark hybrid status" in summary_script
    assert "Benchmark Graph2D status" in summary_script
    assert "Benchmark recommendations" in summary_script
    assert "Benchmark assistant status" in summary_script
    assert "Benchmark review queue status" in summary_script
    assert "Benchmark feedback flywheel status" in summary_script
    assert "Benchmark OCR status" in summary_script
    assert "Benchmark Qdrant status" in summary_script
    assert "Benchmark knowledge readiness status" in summary_script
    assert "Benchmark knowledge reference items" in summary_script
    assert "Benchmark knowledge focus areas" in summary_script
    assert "Benchmark engineering signals status" in summary_script
    assert "Benchmark engineering coverage ratio" in summary_script
    assert "Benchmark engineering standard types" in summary_script
    assert "Benchmark scorecard operator adoption" in summary_script
    assert "Benchmark scorecard operator adoption mode" in summary_script
    assert "Benchmark scorecard operator outcome drift" in summary_script
    assert "Benchmark scorecard operator outcome drift summary" in summary_script
    assert "Feedback flywheel benchmark status" in summary_script
    assert "Feedback flywheel feedback total" in summary_script
    assert "Feedback flywheel artifact" in summary_script
    assert "Benchmark operational summary overall" in summary_script
    assert "Benchmark operational feedback status" in summary_script
    assert "Benchmark operational assistant status" in summary_script
    assert "Benchmark operational review queue status" in summary_script
    assert "Benchmark operational OCR status" in summary_script
    assert "Benchmark operational operator adoption" in summary_script
    assert "Benchmark operational operator outcome drift" in summary_script
    assert "Benchmark operational operator outcome drift summary" in summary_script
    assert "Benchmark operational blockers" in summary_script
    assert "Benchmark operational recommendations" in summary_script
    assert "Benchmark operational artifact" in summary_script
    assert "Benchmark competitive surpass status" in summary_script
    assert "Benchmark competitive surpass score" in summary_script
    assert "Benchmark competitive surpass ready pillars" in summary_script
    assert "Benchmark competitive surpass partial pillars" in summary_script
    assert "Benchmark competitive surpass blocked pillars" in summary_script
    assert "Benchmark competitive surpass primary gaps" in summary_script
    assert "Benchmark competitive surpass recommendations" in summary_script
    assert "Benchmark competitive surpass artifact" in summary_script
    assert "Benchmark artifact bundle overall" in summary_script
    assert "Benchmark artifact bundle available artifacts" in summary_script
    assert "Benchmark artifact bundle feedback status" in summary_script
    assert "Benchmark artifact bundle assistant status" in summary_script
    assert "Benchmark artifact bundle review queue status" in summary_script
    assert "Benchmark artifact bundle OCR status" in summary_script
    assert "Benchmark artifact bundle knowledge status" in summary_script
    assert "Benchmark artifact bundle engineering status" in summary_script
    assert "Benchmark artifact bundle operator adoption knowledge drift" in summary_script
    assert "Benchmark artifact bundle operator adoption knowledge drift summary" in summary_script
    assert "Benchmark artifact bundle operator adoption knowledge outcome drift" in summary_script
    assert (
        "Benchmark artifact bundle operator adoption knowledge outcome drift summary"
        in summary_script
    )
    assert "Benchmark artifact bundle blockers" in summary_script
    assert "Benchmark artifact bundle recommendations" in summary_script
    assert "Benchmark artifact bundle artifact" in summary_script
    assert "Benchmark companion summary overall" in summary_script
    assert "Benchmark companion engineering status" in summary_script
    assert "Benchmark companion review surface" in summary_script
    assert "Benchmark companion primary gap" in summary_script
    assert "Benchmark companion review queue status" in summary_script
    assert "Benchmark companion knowledge status" in summary_script
    assert "Benchmark companion operator adoption knowledge drift" in summary_script
    assert "Benchmark companion operator adoption knowledge drift summary" in summary_script
    assert "Benchmark companion operator adoption knowledge outcome drift" in summary_script
    assert (
        "Benchmark companion operator adoption knowledge outcome drift summary"
        in summary_script
    )
    assert "Benchmark companion recommended actions" in summary_script
    assert "Benchmark companion artifact" in summary_script
    assert "Benchmark release decision status" in summary_script
    assert "Benchmark release automation ready" in summary_script
    assert "Benchmark release primary signal source" in summary_script
    assert "Benchmark release review queue status" in summary_script
    assert "Benchmark release knowledge status" in summary_script
    assert "Benchmark release engineering status" in summary_script
    assert "Benchmark release operator adoption status" in summary_script
    assert "Benchmark release operator adoption knowledge drift" in summary_script
    assert "Benchmark release operator adoption knowledge drift summary" in summary_script
    assert "Benchmark release operator adoption knowledge outcome drift" in summary_script
    assert "Benchmark release operator adoption knowledge outcome drift summary" in summary_script
    assert "Benchmark release review signals" in summary_script
    assert "Benchmark release artifact" in summary_script
    assert "Benchmark release runbook status" in summary_script
    assert "Benchmark release runbook freeze_ready" in summary_script
    assert "Benchmark release runbook primary signal source" in summary_script
    assert "Benchmark release runbook next action" in summary_script
    assert "Benchmark release runbook knowledge status" in summary_script
    assert "Benchmark release runbook engineering status" in summary_script
    assert "Benchmark release runbook operator adoption status" in summary_script
    assert "Benchmark release runbook operator adoption knowledge drift" in summary_script
    assert "Benchmark release runbook operator adoption knowledge drift summary" in summary_script
    assert "Benchmark release runbook operator adoption knowledge outcome drift" in summary_script
    assert (
        "Benchmark release runbook operator adoption knowledge outcome drift summary"
        in summary_script
    )
    assert "Benchmark release runbook missing artifacts" in summary_script
    assert "Benchmark release runbook blocking signals" in summary_script
    assert "Benchmark release runbook review signals" in summary_script
    assert "Benchmark release runbook artifact" in summary_script
    assert "Benchmark operator adoption readiness" in summary_script
    assert "Benchmark operator adoption mode" in summary_script
    assert "Benchmark operator adoption next action" in summary_script
    assert "Benchmark operator adoption automation ready" in summary_script
    assert "Benchmark operator adoption freeze ready" in summary_script
    assert "Benchmark operator adoption release status" in summary_script
    assert "Benchmark operator adoption runbook status" in summary_script
    assert "Benchmark operator adoption review queue status" in summary_script
    assert "Benchmark operator adoption feedback status" in summary_script
    assert "Benchmark operator adoption knowledge drift" in summary_script
    assert "Benchmark operator adoption knowledge drift summary" in summary_script
    assert "Benchmark operator adoption knowledge outcome drift" in summary_script
    assert "Benchmark operator adoption knowledge outcome drift summary" in summary_script
    assert "Benchmark operator adoption release surface alignment" in summary_script
    assert "Benchmark operator adoption release surface alignment summary" in (
        summary_script
    )
    assert "Benchmark operator adoption release surface mismatches" in summary_script
    assert "Benchmark operator adoption blockers" in summary_script
    assert "Benchmark operator adoption actions" in summary_script
    assert "Benchmark operator adoption artifact" in summary_script
    assert "Benchmark engineering signals: skipped" in summary_script
    assert "Benchmark real-data status" in summary_script
    assert "Benchmark real-data ready components" in summary_script
    assert "Benchmark real-data partial components" in summary_script
    assert "Benchmark real-data environment blocked" in summary_script
    assert "Benchmark real-data available components" in summary_script
    assert "Benchmark real-data hybrid status" in summary_script
    assert "Benchmark real-data history status" in summary_script
    assert "Benchmark real-data STEP smoke status" in summary_script
    assert "Benchmark real-data STEP dir status" in summary_script
    assert "Benchmark real-data recommendations" in summary_script
    assert "Benchmark real-data artifact" in summary_script
    assert "Benchmark real-data scorecard status" in summary_script
    assert "Benchmark real-data scorecard ready components" in summary_script
    assert "Benchmark real-data scorecard partial components" in summary_script
    assert "Benchmark real-data scorecard environment blocked" in summary_script
    assert "Benchmark real-data scorecard available components" in summary_script
    assert "Benchmark real-data scorecard best surface" in summary_script
    assert "Benchmark real-data scorecard hybrid status" in summary_script
    assert "Benchmark real-data scorecard history status" in summary_script
    assert "Benchmark real-data scorecard STEP smoke status" in summary_script
    assert "Benchmark real-data scorecard STEP dir status" in summary_script
    assert "Benchmark real-data scorecard recommendations" in summary_script
    assert "Benchmark real-data scorecard artifact" in summary_script
    assert "Benchmark competitive surpass status" in summary_script
    assert "Benchmark competitive surpass score" in summary_script
    assert "Benchmark competitive surpass ready pillars" in summary_script
    assert "Benchmark competitive surpass partial pillars" in summary_script
    assert "Benchmark competitive surpass blocked pillars" in summary_script
    assert "Benchmark competitive surpass primary gaps" in summary_script
    assert "Benchmark competitive surpass recommendations" in summary_script
    assert "Benchmark competitive surpass artifact" in summary_script
    assert "Benchmark engineering violations" in summary_script
    assert "Benchmark engineering standards rows" in summary_script
    assert "Benchmark engineering OCR standards" in summary_script
    assert "Benchmark engineering recommendations" in summary_script
    assert "Benchmark engineering artifact" in summary_script
    assert "Benchmark knowledge readiness: skipped" in summary_script
    assert "Benchmark knowledge total reference items" in summary_script
    assert "Benchmark knowledge ready components" in summary_script
    assert "Benchmark knowledge partial components" in summary_script
    assert "Benchmark knowledge missing components" in summary_script
    assert "Benchmark knowledge focus area count" in summary_script
    assert "Benchmark knowledge domain count" in summary_script
    assert "Benchmark knowledge priority domains" in summary_script
    assert "Benchmark knowledge domain focus areas" in summary_script
    assert "Benchmark knowledge recommendations" in summary_script
    assert "Benchmark knowledge artifact" in summary_script
    assert "Benchmark knowledge drift status" in summary_script
    assert "Benchmark knowledge drift current status" in summary_script
    assert "Benchmark knowledge drift previous status" in summary_script
    assert "Benchmark knowledge drift reference item delta" in summary_script
    assert "Benchmark knowledge drift regressions" in summary_script
    assert "Benchmark knowledge drift improvements" in summary_script
    assert "Benchmark knowledge drift resolved focus areas" in summary_script
    assert "Benchmark knowledge drift new focus areas" in summary_script
    assert "Benchmark knowledge drift recommendations" in summary_script
    assert "Benchmark knowledge drift artifact" in summary_script
    assert "Benchmark knowledge application status" in summary_script
    assert "Benchmark knowledge application ready domains" in summary_script
    assert "Benchmark knowledge application partial domains" in summary_script
    assert "Benchmark knowledge application missing domains" in summary_script
    assert "Benchmark knowledge application total domains" in summary_script
    assert "Benchmark knowledge application focus area count" in summary_script
    assert "Benchmark knowledge application focus areas" in summary_script
    assert "Benchmark knowledge application priority domains" in summary_script
    assert "Benchmark knowledge application domain statuses" in summary_script
    assert "Benchmark knowledge application recommendations" in summary_script
    assert "Benchmark knowledge application artifact" in summary_script
    assert "Benchmark knowledge realdata correlation status" in summary_script
    assert "Benchmark knowledge realdata correlation ready domains" in summary_script
    assert "Benchmark knowledge realdata correlation partial domains" in summary_script
    assert "Benchmark knowledge realdata correlation blocked domains" in summary_script
    assert "Benchmark knowledge realdata correlation total domains" in summary_script
    assert "Benchmark knowledge realdata correlation focus areas" in summary_script
    assert "Benchmark knowledge realdata correlation priority domains" in summary_script
    assert "Benchmark knowledge realdata correlation domain statuses" in summary_script
    assert "Benchmark knowledge realdata correlation recommendations" in summary_script
    assert "Benchmark knowledge realdata correlation artifact" in summary_script
    assert "Benchmark knowledge domain action plan status" in summary_script
    assert "Benchmark knowledge domain action plan ready domains" in summary_script
    assert "Benchmark knowledge domain action plan partial domains" in summary_script
    assert "Benchmark knowledge domain action plan blocked domains" in summary_script
    assert "Benchmark knowledge domain action plan total domains" in summary_script
    assert "Benchmark knowledge domain action plan total actions" in summary_script
    assert "Benchmark knowledge domain action plan high-priority actions" in summary_script
    assert "Benchmark knowledge domain action plan medium-priority actions" in summary_script
    assert "Benchmark knowledge domain action plan priority domains" in summary_script
    assert "Benchmark knowledge domain action plan first actions" in summary_script
    assert "Benchmark knowledge domain action plan domain action counts" in summary_script
    assert "Benchmark knowledge domain action plan recommendations" in summary_script
    assert "Benchmark knowledge domain action plan artifact" in summary_script
    assert "Benchmark knowledge source action plan status" in summary_script
    assert "Benchmark knowledge source action plan total actions" in summary_script
    assert "Benchmark knowledge source action plan high-priority actions" in (
        summary_script
    )
    assert "Benchmark knowledge source action plan medium-priority actions" in (
        summary_script
    )
    assert "Benchmark knowledge source action plan expansion actions" in summary_script
    assert "Benchmark knowledge source action plan priority domains" in summary_script
    assert "Benchmark knowledge source action plan first actions" in summary_script
    assert "Benchmark knowledge source action plan source-group counts" in (
        summary_script
    )
    assert "Benchmark knowledge source action plan recommendations" in summary_script
    assert "Benchmark knowledge source action plan artifact" in summary_script
    assert "Benchmark artifact bundle knowledge drift" in summary_script
    assert "Benchmark artifact bundle knowledge drift summary" in summary_script
    assert "Benchmark artifact bundle knowledge drift recommendations" in summary_script
    assert "Benchmark artifact bundle knowledge drift component changes" in summary_script
    assert "Benchmark artifact bundle knowledge focus areas" in summary_script
    assert "Benchmark artifact bundle knowledge application" in summary_script
    assert "Benchmark artifact bundle knowledge application focus areas" in summary_script
    assert "Benchmark artifact bundle knowledge application domains" in summary_script
    assert "Benchmark artifact bundle knowledge application recommendations" in summary_script
    assert "Benchmark artifact bundle knowledge realdata correlation" in summary_script
    assert "Benchmark artifact bundle knowledge realdata focus areas" in summary_script
    assert "Benchmark artifact bundle knowledge realdata domains" in summary_script
    assert "Benchmark artifact bundle knowledge realdata recommendations" in summary_script
    assert "Benchmark artifact bundle knowledge source action plan" in summary_script
    assert "Benchmark artifact bundle knowledge source action plan domains" in (
        summary_script
    )
    assert "Benchmark artifact bundle knowledge source action plan first actions" in (
        summary_script
    )
    assert "Benchmark artifact bundle knowledge source action plan source-group counts" in (
        summary_script
    )
    assert (
        "Benchmark artifact bundle knowledge source action plan recommendations"
        in summary_script
    )
    assert "Benchmark artifact bundle real-data status" in summary_script
    assert "Benchmark artifact bundle real-data recommendations" in summary_script
    assert "Benchmark artifact bundle competitive surpass" in summary_script
    assert "Benchmark artifact bundle competitive surpass gaps" in summary_script
    assert "Benchmark artifact bundle competitive surpass recommendations" in summary_script
    assert "Benchmark artifact bundle scorecard operator adoption" in summary_script
    assert "Benchmark artifact bundle scorecard operator outcome drift" in summary_script
    assert "Benchmark artifact bundle operational operator adoption" in summary_script
    assert "Benchmark artifact bundle operational operator outcome drift" in summary_script
    assert "Benchmark artifact bundle operator adoption release surface alignment" in (
        summary_script
    )
    assert "Benchmark artifact bundle operator adoption release surface mismatches" in (
        summary_script
    )
    assert "Benchmark companion knowledge drift" in summary_script
    assert "Benchmark companion knowledge drift summary" in summary_script
    assert "Benchmark companion knowledge drift recommendations" in summary_script
    assert "Benchmark companion knowledge drift component changes" in summary_script
    assert "Benchmark companion knowledge focus areas" in summary_script
    assert "Benchmark companion knowledge application" in summary_script
    assert "Benchmark companion knowledge application focus areas" in summary_script
    assert "Benchmark companion knowledge application domains" in summary_script
    assert "Benchmark companion knowledge application recommendations" in summary_script
    assert "Benchmark companion knowledge realdata correlation" in summary_script
    assert "Benchmark companion knowledge realdata focus areas" in summary_script
    assert "Benchmark companion knowledge realdata domains" in summary_script
    assert "Benchmark companion knowledge realdata recommendations" in summary_script
    assert "Benchmark companion knowledge source action plan" in summary_script
    assert "Benchmark companion knowledge source action plan domains" in summary_script
    assert "Benchmark companion knowledge source action plan first actions" in (
        summary_script
    )
    assert "Benchmark companion knowledge source action plan source-group counts" in (
        summary_script
    )
    assert "Benchmark companion knowledge source action plan recommendations" in (
        summary_script
    )
    assert "Benchmark companion real-data status" in summary_script
    assert "Benchmark companion real-data recommendations" in summary_script
    assert "Benchmark companion competitive surpass" in summary_script
    assert "Benchmark companion competitive surpass gaps" in summary_script
    assert "Benchmark companion competitive surpass recommendations" in summary_script
    assert "Benchmark companion scorecard operator adoption" in summary_script
    assert "Benchmark companion scorecard operator outcome drift" in summary_script
    assert "Benchmark companion operational operator adoption" in summary_script
    assert "Benchmark companion operational operator outcome drift" in summary_script
    assert "Benchmark companion operator adoption release surface alignment" in (
        summary_script
    )
    assert "Benchmark companion operator adoption release surface mismatches" in (
        summary_script
    )
    assert "Benchmark release knowledge drift" in summary_script
    assert "Benchmark release knowledge drift summary" in summary_script
    assert "Benchmark release knowledge focus areas" in summary_script
    assert "Benchmark release knowledge application" in summary_script
    assert "Benchmark release knowledge application focus areas" in summary_script
    assert "Benchmark release knowledge application domains" in summary_script
    assert "Benchmark release knowledge application recommendations" in summary_script
    assert "Benchmark release knowledge realdata correlation" in summary_script
    assert "Benchmark release knowledge realdata focus areas" in summary_script
    assert "Benchmark release knowledge realdata domains" in summary_script
    assert "Benchmark release knowledge realdata recommendations" in summary_script
    assert "Benchmark release knowledge source action plan" in summary_script
    assert "Benchmark release knowledge source action plan domains" in summary_script
    assert "Benchmark release knowledge source action plan first actions" in (
        summary_script
    )
    assert "Benchmark release knowledge source action plan source-group counts" in (
        summary_script
    )
    assert "Benchmark release knowledge source action plan recommendations" in (
        summary_script
    )
    assert "Benchmark release real-data status" in summary_script
    assert "Benchmark release real-data recommendations" in summary_script
    assert "Benchmark release runbook knowledge drift" in summary_script
    assert "Benchmark release runbook knowledge drift summary" in summary_script
    assert "Benchmark release runbook knowledge focus areas" in summary_script
    assert "Benchmark release runbook knowledge application" in summary_script
    assert "Benchmark release runbook knowledge application focus areas" in summary_script
    assert "Benchmark release runbook knowledge application domains" in summary_script
    assert "Benchmark release runbook knowledge application recommendations" in summary_script
    assert "Benchmark release runbook knowledge realdata correlation" in summary_script
    assert "Benchmark release runbook knowledge realdata focus areas" in summary_script
    assert "Benchmark release runbook knowledge realdata domains" in summary_script
    assert "Benchmark release runbook knowledge realdata recommendations" in summary_script
    assert "Benchmark release runbook knowledge source action plan" in summary_script
    assert "Benchmark release runbook knowledge source action plan domains" in (
        summary_script
    )
    assert "Benchmark release runbook knowledge source action plan first actions" in (
        summary_script
    )
    assert (
        "Benchmark release runbook knowledge source action plan source-group counts"
        in summary_script
    )
    assert (
        "Benchmark release runbook knowledge source action plan recommendations"
        in summary_script
    )
    assert "Benchmark release runbook real-data status" in summary_script
    assert "Benchmark release runbook real-data recommendations" in summary_script
    assert "Assistant evidence input" in summary_script
    assert "Assistant evidence records" in summary_script
    assert "Assistant evidence items" in summary_script
    assert "Assistant record kinds" in summary_script
    assert "Assistant evidence types" in summary_script
    assert "Assistant missing fields" in summary_script
    assert "Active-learning review queue input" in summary_script
    assert "Active-learning review queue total" in summary_script
    assert "Active-learning review queue status" in summary_script
    assert "Active-learning review queue priorities" in summary_script
    assert "Active-learning review queue decision sources" in summary_script
    assert "Active-learning review queue review reasons" in summary_script
    assert "OCR review pack input" in summary_script
    assert "OCR review pack exported" in summary_script
    assert "OCR review priorities" in summary_script
    assert "OCR recommended actions" in summary_script

    pr_comment_step = _get_step(workflow, "evaluate", "Comment PR with results")
    pr_comment_script = pr_comment_step["with"]["script"]
    assert "reviewInputCsv" in pr_comment_script
    assert "reviewInputSource" in pr_comment_script
    assert "Graph2D Review Gate" in pr_comment_script
    assert "Graph2D Review Gate Strict" in pr_comment_script
    assert "Graph2D Train Sweep" in pr_comment_script
    assert "Graph2D Review Insights" in pr_comment_script
    assert "knowledge=" in pr_comment_script
    assert "priorities=" in pr_comment_script
    assert "bands=" in pr_comment_script
    assert "Graph2D Signal Lights" in pr_comment_script
    assert "reviewTopShadowSources" in pr_comment_script
    assert "script=${sweepBestRunScript}" in pr_comment_script
    assert "benchmarkScorecardEnabled" in pr_comment_script
    assert "Benchmark Scorecard" in pr_comment_script
    assert "benchmarkScorecardOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkScorecardOperatorAdoptionMode" in pr_comment_script
    assert "benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus" in pr_comment_script
    assert "benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary" in pr_comment_script
    assert "Benchmark Recommendations" in pr_comment_script
    assert "benchmarkAssistantStatus" in pr_comment_script
    assert "benchmarkReviewQueueStatus" in pr_comment_script
    assert "benchmarkFeedbackFlywheelStatus" in pr_comment_script
    assert "feedbackFlywheelBenchmarkEnabled" in pr_comment_script
    assert "feedbackFlywheelBenchmarkStatus" in pr_comment_script
    assert "feedbackFlywheelBenchmarkArtifact" in pr_comment_script
    assert "benchmarkOperationalSummaryEnabled" in pr_comment_script
    assert "benchmarkOperationalSummaryOverall" in pr_comment_script
    assert "benchmarkOperationalSummaryStatus" in pr_comment_script
    assert "benchmarkOperationalOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus" in pr_comment_script
    assert "benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary" in pr_comment_script
    assert "benchmarkOperationalLight" in pr_comment_script
    assert "benchmarkArtifactBundleEnabled" in pr_comment_script
    assert "benchmarkArtifactBundleOverall" in pr_comment_script
    assert "benchmarkArtifactBundleAvailableArtifacts" in pr_comment_script
    assert "benchmarkArtifactBundleStatus" in pr_comment_script
    assert "benchmarkArtifactBundleLight" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDriftRecommendations" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDriftChanges" in pr_comment_script
    assert "benchmarkArtifactBundleEngineeringStatus" in pr_comment_script
    assert "benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkArtifactBundleScorecardOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkArtifactBundleScorecardOperatorAdoptionMode" in pr_comment_script
    assert "benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentStatus" in (
        pr_comment_script
    )
    assert "benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentMismatches" in (
        pr_comment_script
    )
    assert (
        "benchmarkArtifactBundleScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus"
        in pr_comment_script
    )
    assert "benchmarkArtifactBundleOperationalOperatorAdoptionStatus" in pr_comment_script
    assert (
        "benchmarkArtifactBundleOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus"
        in pr_comment_script
    )
    assert "benchmarkCompanionSummaryEnabled" in pr_comment_script
    assert "benchmarkCompanionSummaryOverall" in pr_comment_script
    assert "benchmarkCompanionSummaryStatus" in pr_comment_script
    assert "benchmarkCompanionLight" in pr_comment_script
    assert "benchmarkCompanionReviewSurface" in pr_comment_script
    assert "benchmarkCompanionPrimaryGap" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDriftRecommendations" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDriftChanges" in pr_comment_script
    assert "benchmarkCompanionEngineeringStatus" in pr_comment_script
    assert "benchmarkCompanionOperatorAdoptionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkCompanionOperatorAdoptionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkCompanionScorecardOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkCompanionScorecardOperatorAdoptionMode" in pr_comment_script
    assert "benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentStatus" in (
        pr_comment_script
    )
    assert "benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentMismatches" in (
        pr_comment_script
    )
    assert (
        "benchmarkCompanionScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus"
        in pr_comment_script
    )
    assert "benchmarkCompanionOperationalOperatorAdoptionStatus" in pr_comment_script
    assert (
        "benchmarkCompanionOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus"
        in pr_comment_script
    )
    assert "benchmarkReleaseDecisionEnabled" in pr_comment_script
    assert "benchmarkReleaseStatus" in pr_comment_script
    assert "benchmarkReleaseDecisionStatus" in pr_comment_script
    assert "benchmarkReleaseDecisionLight" in pr_comment_script
    assert "benchmarkReleasePrimarySignalSource" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkReleaseEngineeringStatus" in pr_comment_script
    assert "benchmarkReleaseOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkReleaseOperatorAdoptionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkReleaseOperatorAdoptionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkReleaseRunbookEnabled" in pr_comment_script
    assert "benchmarkReleaseRunbookStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookLight" in pr_comment_script
    assert "benchmarkReleaseRunbookNextAction" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkReleaseRunbookEngineeringStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookOperatorAdoptionStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkOperatorAdoptionEnabled" in pr_comment_script
    assert "benchmarkOperatorAdoptionReadiness" in pr_comment_script
    assert "benchmarkOperatorAdoptionStatusLine" in pr_comment_script
    assert "benchmarkOperatorAdoptionLight" in pr_comment_script
    assert "benchmarkOperatorAdoptionNextAction" in pr_comment_script
    assert "benchmarkOperatorAdoptionKnowledgeDriftStatus" in pr_comment_script
    assert "benchmarkOperatorAdoptionKnowledgeDriftSummary" in pr_comment_script
    assert "benchmarkOcrStatus" in pr_comment_script
    assert "benchmarkQdrantStatus" in pr_comment_script
    assert "benchmarkKnowledgeStatus" in pr_comment_script
    assert "benchmarkEngineeringEnabled" in pr_comment_script
    assert "benchmarkEngineeringStatus" in pr_comment_script
    assert "benchmarkEngineeringTopStandardTypes" in pr_comment_script
    assert "benchmarkEngineeringStatusLine" in pr_comment_script
    assert "benchmarkEngineeringLight" in pr_comment_script
    assert "benchmarkRealdataEnabled" in pr_comment_script
    assert "benchmarkRealdataStatus" in pr_comment_script
    assert "benchmarkRealdataStatusLine" in pr_comment_script
    assert "benchmarkRealdataLight" in pr_comment_script
    assert "benchmarkRealdataRecommendations" in pr_comment_script
    assert "benchmarkRealdataScorecardEnabled" in pr_comment_script
    assert "benchmarkRealdataScorecardStatus" in pr_comment_script
    assert "benchmarkRealdataScorecardStatusLine" in pr_comment_script
    assert "benchmarkRealdataScorecardLight" in pr_comment_script
    assert "benchmarkRealdataScorecardRecommendations" in pr_comment_script
    assert "benchmarkCompetitiveSurpassEnabled" in pr_comment_script
    assert "benchmarkCompetitiveSurpassStatus" in pr_comment_script
    assert "benchmarkCompetitiveSurpassStatusLine" in pr_comment_script
    assert "benchmarkCompetitiveSurpassLight" in pr_comment_script
    assert "benchmarkCompetitiveSurpassRecommendations" in pr_comment_script
    assert "Benchmark Competitive Surpass" in pr_comment_script
    assert "benchmarkKnowledgeEnabled" in pr_comment_script
    assert "benchmarkKnowledgeStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeLight" in pr_comment_script
    assert "benchmarkKnowledgeDriftEnabled" in pr_comment_script
    assert "benchmarkKnowledgeDriftStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeDriftLight" in pr_comment_script
    assert "benchmarkKnowledgeDriftDomainRegressions" in pr_comment_script
    assert "benchmarkKnowledgeDriftResolvedPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeApplicationEnabled" in pr_comment_script
    assert "benchmarkKnowledgeApplicationStatus" in pr_comment_script
    assert "benchmarkKnowledgeApplicationStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeApplicationLight" in pr_comment_script
    assert "benchmarkKnowledgeApplicationFocusAreas" in pr_comment_script
    assert "benchmarkKnowledgeApplicationPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeApplicationDomainStatuses" in pr_comment_script
    assert "benchmarkKnowledgeApplicationRecommendations" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationEnabled" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationStatus" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationLight" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationFocusAreas" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationDomainStatuses" in pr_comment_script
    assert "benchmarkKnowledgeRealdataCorrelationRecommendations" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixEnabled" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixStatus" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixLight" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixFocusAreas" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixDomainStatuses" in pr_comment_script
    assert "benchmarkKnowledgeDomainMatrixRecommendations" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanEnabled" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanStatus" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanLight" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanRecommendedFirstActions" in pr_comment_script
    assert "benchmarkKnowledgeDomainActionPlanRecommendations" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanEnabled" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanStatus" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanLight" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeSourceActionPlanRecommendedFirstActions" in (
        pr_comment_script
    )
    assert "benchmarkKnowledgeSourceActionPlanRecommendations" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationEnabled" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationStatus" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationStatusLine" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationLight" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationFocusAreas" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationPriorityDomains" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationDomainStatuses" in pr_comment_script
    assert "benchmarkKnowledgeOutcomeCorrelationRecommendations" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeApplicationStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeApplicationStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleCompetitiveSurpassStatus" in pr_comment_script
    assert "benchmarkArtifactBundleCompetitiveSurpassStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleCompetitiveSurpassLight" in pr_comment_script
    assert "Benchmark Artifact Bundle Competitive Surpass" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeRealdataCorrelationStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeRealdataCorrelationStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDomainMatrixStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDomainMatrixStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDomainActionPlanStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDomainActionPlanStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeSourceActionPlanStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeSourceActionPlanStatusLine" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatus" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatusLine" in pr_comment_script
    assert "benchmarkCompanionKnowledgeApplicationStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeApplicationStatusLine" in pr_comment_script
    assert "benchmarkCompanionCompetitiveSurpassStatus" in pr_comment_script
    assert "benchmarkCompanionCompetitiveSurpassStatusLine" in pr_comment_script
    assert "benchmarkCompanionCompetitiveSurpassLight" in pr_comment_script
    assert "Benchmark Companion Competitive Surpass" in pr_comment_script
    assert "benchmarkCompanionKnowledgeRealdataCorrelationStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeRealdataCorrelationStatusLine" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDomainMatrixStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDomainMatrixStatusLine" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDomainActionPlanStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDomainActionPlanStatusLine" in pr_comment_script
    assert "benchmarkCompanionKnowledgeSourceActionPlanStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeSourceActionPlanStatusLine" in pr_comment_script
    assert "benchmarkCompanionKnowledgeOutcomeCorrelationStatus" in pr_comment_script
    assert "benchmarkCompanionKnowledgeOutcomeCorrelationStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeApplicationStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeApplicationStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeRealdataCorrelationStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeRealdataCorrelationStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDomainMatrixStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDomainMatrixStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDomainActionPlanStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDomainActionPlanStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeSourceActionPlanStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeSourceActionPlanStatusLine" in pr_comment_script
    assert "benchmarkReleaseKnowledgeOutcomeCorrelationStatus" in pr_comment_script
    assert "benchmarkReleaseKnowledgeOutcomeCorrelationStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeApplicationStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeApplicationStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDomainMatrixStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDomainMatrixStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDomainActionPlanStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDomainActionPlanStatusLine" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeSourceActionPlanStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeSourceActionPlanStatusLine" in (
        pr_comment_script
    )
    assert "benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatus" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatusLine" in pr_comment_script
    assert "assistant=${benchmarkAssistantStatus}" in pr_comment_script
    assert "review_queue=${benchmarkReviewQueueStatus}" in pr_comment_script
    assert "feedback_flywheel=${benchmarkFeedbackFlywheelStatus}" in pr_comment_script
    assert "ocr=${benchmarkOcrStatus}" in pr_comment_script
    assert "qdrant=${benchmarkQdrantStatus}" in pr_comment_script
    assert "knowledge=${benchmarkKnowledgeStatus}" in pr_comment_script
    assert "engineering=${benchmarkEngineeringStatus}" in pr_comment_script
    assert "Benchmark Feedback Flywheel" in pr_comment_script
    assert "Benchmark Engineering Signals" in pr_comment_script
    assert "Benchmark Scorecard Operator Adoption" in pr_comment_script
    assert "Benchmark Scorecard Operator Outcome Drift" in pr_comment_script
    assert "Benchmark Real-Data Signals" in pr_comment_script
    assert "Benchmark Real-Data Scorecard" in pr_comment_script
    assert "Benchmark Knowledge Readiness" in pr_comment_script
    assert "Benchmark Knowledge Drift" in pr_comment_script
    assert "Benchmark Knowledge Focus Areas" in pr_comment_script
    assert "Benchmark Knowledge Domains" in pr_comment_script
    assert "Benchmark Knowledge Recommendations" in pr_comment_script
    assert "Benchmark Knowledge Application" in pr_comment_script
    assert "Benchmark Knowledge Application Recommendations" in pr_comment_script
    assert "Benchmark Knowledge Real-Data Correlation" in pr_comment_script
    assert "Benchmark Knowledge Domain Matrix" in pr_comment_script
    assert "Benchmark Knowledge Domain Action Plan" in pr_comment_script
    assert "Benchmark Knowledge Source Action Plan" in pr_comment_script
    assert "Benchmark Knowledge Outcome Correlation" in pr_comment_script
    assert "Benchmark Knowledge Real-Data Recommendations" in pr_comment_script
    assert "Benchmark Knowledge Outcome Recommendations" in pr_comment_script
    assert "Benchmark Knowledge Drift Recommendations" in pr_comment_script
    assert "Benchmark Engineering Recommendations" in pr_comment_script
    assert "Feedback Flywheel Artifact" in pr_comment_script
    assert "Benchmark Operational Summary" in pr_comment_script
    assert "Benchmark Operational Operator Adoption" in pr_comment_script
    assert "Benchmark Operational Operator Outcome Drift" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Drift" in pr_comment_script
    assert "Benchmark Artifact Bundle" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Application" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Real-Data" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Domain Matrix" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Domain Action Plan" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Source Action Plan" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Outcome Correlation" in pr_comment_script
    assert "Benchmark Artifact Bundle Real-Data" in pr_comment_script
    assert "Benchmark Artifact Bundle Real-Data Scorecard" in pr_comment_script
    assert "available_artifacts=${benchmarkArtifactBundleAvailableArtifacts}" in pr_comment_script
    assert "feedback=${benchmarkArtifactBundleFeedbackStatus}" in pr_comment_script
    assert "assistant=${benchmarkArtifactBundleAssistantStatus}" in pr_comment_script
    assert "review_queue=${benchmarkArtifactBundleReviewQueueStatus}" in pr_comment_script
    assert "ocr=${benchmarkArtifactBundleOcrStatus}" in pr_comment_script
    assert "knowledge=${benchmarkArtifactBundleKnowledgeStatus}" in pr_comment_script
    assert "knowledge_drift=${benchmarkArtifactBundleKnowledgeDriftStatus}" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgeDriftDomainRegressions" in pr_comment_script
    assert "benchmarkArtifactBundleKnowledgePriorityDomains" in pr_comment_script
    assert "Benchmark Companion Knowledge Application" in pr_comment_script
    assert "Benchmark Companion Knowledge Real-Data" in pr_comment_script
    assert "Benchmark Companion Knowledge Domain Matrix" in pr_comment_script
    assert "Benchmark Companion Knowledge Domain Action Plan" in pr_comment_script
    assert "Benchmark Companion Knowledge Source Action Plan" in pr_comment_script
    assert "Benchmark Companion Knowledge Outcome Correlation" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Application" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Real-Data" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Domain Matrix" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Domain Action Plan" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Source Action Plan" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Outcome Correlation" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Application" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Real-Data" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Domain Matrix" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Domain Action Plan" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Source Action Plan" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Outcome Correlation" in pr_comment_script
    assert (
        "recommendations=${benchmarkKnowledgeApplicationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkKnowledgeRealdataCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkKnowledgeDomainMatrixRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkKnowledgeDomainActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkKnowledgeSourceActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkKnowledgeOutcomeCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeApplicationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeRealdataCorrelation"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeDomainMatrixRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeDomainActionPlan"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeSourceActionPlan"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkArtifactBundleKnowledgeOutcomeCorrelation"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeApplicationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeRealdataCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeDomainMatrixRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeDomainActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeSourceActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkCompanionKnowledgeOutcomeCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeApplicationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeRealdataCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeDomainMatrixRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeDomainActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeSourceActionPlanRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseKnowledgeOutcomeCorrelationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeApplicationRecommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeRealdataCorrelation"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeDomainMatrix"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeDomainActionPlan"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeSourceActionPlan"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert (
        "recommendations=${benchmarkReleaseRunbookKnowledgeOutcomeCorrelation"
        "Recommendations || 'n/a'}"
        in pr_comment_script
    )
    assert "engineering=${benchmarkArtifactBundleEngineeringStatus}" in pr_comment_script
    assert "Benchmark Artifact Bundle Knowledge Domains" in pr_comment_script
    assert "Benchmark Companion Knowledge Drift" in pr_comment_script
    assert (
        "operator_drift=${benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftStatus}"
        in pr_comment_script
    )
    assert "Benchmark Companion Summary" in pr_comment_script
    assert "Benchmark Companion Real-Data" in pr_comment_script
    assert "Benchmark Companion Real-Data Scorecard" in pr_comment_script
    assert "knowledge=${benchmarkCompanionKnowledgeStatus}" in pr_comment_script
    assert "knowledge_drift=${benchmarkCompanionKnowledgeDriftStatus}" in pr_comment_script
    assert "benchmarkCompanionKnowledgeDriftDomainRegressions" in pr_comment_script
    assert "benchmarkCompanionKnowledgePriorityDomains" in pr_comment_script
    assert "engineering=${benchmarkCompanionEngineeringStatus}" in pr_comment_script
    assert "Benchmark Companion Knowledge Domains" in pr_comment_script
    assert "Benchmark Release Decision Knowledge Drift" in pr_comment_script
    assert (
        "operator_drift=${benchmarkCompanionOperatorAdoptionKnowledgeDriftStatus}"
        in pr_comment_script
    )
    assert "Benchmark Release Decision" in pr_comment_script
    assert "Benchmark Release Decision Real-Data" in pr_comment_script
    assert "Benchmark Release Decision Real-Data Scorecard" in pr_comment_script
    assert "Benchmark Release Decision Scorecard Operator Adoption" in pr_comment_script
    assert "Benchmark Release Decision Scorecard Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Release Decision Operational Operator Adoption" in (
        pr_comment_script
    )
    assert "Benchmark Release Decision Operational Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Release Decision Release Surface Alignment" in (
        pr_comment_script
    )
    assert "knowledge=${benchmarkReleaseKnowledgeStatus}" in pr_comment_script
    assert "knowledge_drift=${benchmarkReleaseKnowledgeDriftStatus}" in pr_comment_script
    assert "benchmarkReleaseKnowledgeDriftDomainRegressions" in pr_comment_script
    assert "benchmarkReleaseKnowledgePriorityDomains" in pr_comment_script
    assert "engineering=${benchmarkReleaseEngineeringStatus}" in pr_comment_script
    assert "operator_adoption=${benchmarkReleaseOperatorAdoptionStatus}" in pr_comment_script
    assert (
        "benchmarkReleaseScorecardOperatorAdoptionStatus || 'n/a'" in pr_comment_script
    )
    assert (
        "benchmarkReleaseOperationalOperatorAdoptionStatus || 'n/a'" in pr_comment_script
    )
    assert (
        "benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'"
        in pr_comment_script
    )
    assert "Benchmark Release Decision Knowledge Domains" in pr_comment_script
    assert "Benchmark Release Runbook Knowledge Drift" in pr_comment_script
    assert (
        "operator_drift=${benchmarkReleaseOperatorAdoptionKnowledgeDriftStatus}"
        in pr_comment_script
    )
    assert "Benchmark Release Runbook" in pr_comment_script
    assert "Benchmark Release Runbook Real-Data" in pr_comment_script
    assert "Benchmark Release Runbook Real-Data Scorecard" in pr_comment_script
    assert "Benchmark Release Runbook Scorecard Operator Adoption" in pr_comment_script
    assert "Benchmark Release Runbook Scorecard Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Release Runbook Operational Operator Adoption" in (
        pr_comment_script
    )
    assert "Benchmark Release Runbook Operational Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Release Runbook Release Surface Alignment" in (
        pr_comment_script
    )
    assert "knowledge=${benchmarkReleaseRunbookKnowledgeStatus}" in pr_comment_script
    assert "knowledge_drift=${benchmarkReleaseRunbookKnowledgeDriftStatus}" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgeDriftDomainRegressions" in pr_comment_script
    assert "benchmarkReleaseRunbookKnowledgePriorityDomains" in pr_comment_script
    assert "engineering=${benchmarkReleaseRunbookEngineeringStatus}" in pr_comment_script
    assert "operator_adoption=${benchmarkReleaseRunbookOperatorAdoptionStatus}" in pr_comment_script
    assert (
        "benchmarkReleaseRunbookScorecardOperatorAdoptionStatus || 'n/a'"
        in pr_comment_script
    )
    assert (
        "benchmarkReleaseRunbookOperationalOperatorAdoptionStatus || 'n/a'"
        in pr_comment_script
    )
    assert (
        "benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'"
        in pr_comment_script
    )
    assert (
        "operator_drift=${benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftStatus}"
        in pr_comment_script
    )
    assert "Benchmark Release Runbook Knowledge Domains" in pr_comment_script
    assert "Benchmark Operator Adoption" in pr_comment_script
    assert "automation_ready=${benchmarkReleaseAutomationReady}" in pr_comment_script
    assert "source=${benchmarkReleasePrimarySignalSource}" in pr_comment_script
    assert "next=${benchmarkReleaseRunbookNextAction || 'n/a'}" in pr_comment_script
    assert "mode=${benchmarkOperatorAdoptionMode}" in pr_comment_script
    assert "readiness=${benchmarkOperatorAdoptionReadiness}" in pr_comment_script
    assert "knowledge_drift=${benchmarkOperatorAdoptionKnowledgeDriftStatus}" in pr_comment_script
    assert "Benchmark Operator Adoption Release Surface Alignment" in pr_comment_script
    assert "Benchmark Operator Adoption Release Surface Mismatches" in (
        pr_comment_script
    )
    assert (
        "benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'"
        in pr_comment_script
    )
    assert (
        "benchmarkOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'"
        in pr_comment_script
    )
    assert "Benchmark Artifact Bundle Operator Drift" in pr_comment_script
    assert "Benchmark Artifact Bundle Operator Outcome Drift" in pr_comment_script
    assert "Benchmark Artifact Bundle Scorecard Operator Adoption" in pr_comment_script
    assert "Benchmark Artifact Bundle Scorecard Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Artifact Bundle Operational Operator Adoption" in pr_comment_script
    assert "Benchmark Artifact Bundle Operational Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Artifact Bundle Release Surface Alignment" in pr_comment_script
    assert "Benchmark Artifact Bundle Release Surface Mismatches" in pr_comment_script
    assert "Benchmark Companion Operator Drift" in pr_comment_script
    assert "Benchmark Companion Operator Outcome Drift" in pr_comment_script
    assert "Benchmark Companion Scorecard Operator Adoption" in pr_comment_script
    assert "Benchmark Companion Scorecard Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Companion Operational Operator Adoption" in pr_comment_script
    assert "Benchmark Companion Operational Operator Outcome Drift" in (
        pr_comment_script
    )
    assert "Benchmark Companion Release Surface Alignment" in pr_comment_script
    assert "Benchmark Companion Release Surface Mismatches" in pr_comment_script
    assert "Benchmark Release Decision Operator Drift" in pr_comment_script
    assert "Benchmark Release Runbook Operator Drift" in pr_comment_script
    assert "Benchmark Operator Adoption Knowledge Drift" in pr_comment_script
    assert "realdata=${benchmarkArtifactBundleRealdataStatus || 'n/a'}" in pr_comment_script
    assert "benchmarkArtifactBundleRealdataScorecardStatus || 'n/a'" in pr_comment_script
    assert "realdata=${benchmarkCompanionRealdataStatus || 'n/a'}" in pr_comment_script
    assert "benchmarkCompanionRealdataScorecardStatus || 'n/a'" in pr_comment_script
    assert "realdata=${benchmarkReleaseRealdataStatus || 'n/a'}" in pr_comment_script
    assert "benchmarkReleaseRealdataScorecardStatus || 'n/a'" in pr_comment_script
    assert "realdata=${benchmarkReleaseRunbookRealdataStatus || 'n/a'}" in pr_comment_script
    assert "benchmarkReleaseRunbookRealdataScorecardStatus || 'n/a'" in pr_comment_script
    assert "assistantEvidenceEnabled" in pr_comment_script
    assert "Assistant Evidence Report" in pr_comment_script
    assert "Assistant Evidence Insights" in pr_comment_script
    assert "activeLearningReviewQueueEnabled" in pr_comment_script
    assert "Active-Learning Review Queue" in pr_comment_script
    assert "Active-Learning Review Queue Insights" in pr_comment_script
    assert "activeLearningReviewQueueTopDecisionSources" in pr_comment_script
    assert "ocrReviewPackEnabled" in pr_comment_script
    assert "OCR Review Pack" in pr_comment_script
    assert "OCR Review Insights" in pr_comment_script
