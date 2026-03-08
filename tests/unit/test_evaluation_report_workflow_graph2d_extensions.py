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
    assert "BENCHMARK_SCORECARD_OUTPUT_JSON" in env
    assert "BENCHMARK_SCORECARD_OUTPUT_MD" in env
    assert "FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_JSON" in env
    assert "FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_MD" in env
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
    assert "overall_status=" in benchmark_script
    assert "assistant_status=" in benchmark_script
    assert "review_queue_status=" in benchmark_script
    assert "feedback_flywheel_status=" in benchmark_script
    assert "ocr_status=" in benchmark_script
    assert "qdrant_status=" in benchmark_script

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


def test_workflow_uploads_new_graph2d_artifacts_and_summary_lines() -> None:
    workflow = _load_workflow()

    upload_review = _get_step(workflow, "evaluate", "Upload Graph2D review pack")
    assert upload_review["if"] == "steps.graph2d_review_pack.outputs.enabled == 'true'"

    upload_sweep = _get_step(workflow, "evaluate", "Upload Graph2D train sweep")
    assert upload_sweep["if"] == "steps.graph2d_train_sweep.outputs.enabled == 'true'"

    upload_scorecard = _get_step(workflow, "evaluate", "Upload benchmark scorecard")
    assert upload_scorecard["if"] == "steps.benchmark_scorecard.outputs.enabled == 'true'"
    upload_feedback_flywheel = _get_step(
        workflow, "evaluate", "Upload feedback flywheel benchmark artifact"
    )
    assert (
        upload_feedback_flywheel["if"]
        == "steps.feedback_flywheel_benchmark.outputs.enabled == 'true'"
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
    assert "Feedback flywheel benchmark status" in summary_script
    assert "Feedback flywheel feedback total" in summary_script
    assert "Feedback flywheel artifact" in summary_script
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
    assert "Benchmark Recommendations" in pr_comment_script
    assert "benchmarkAssistantStatus" in pr_comment_script
    assert "benchmarkReviewQueueStatus" in pr_comment_script
    assert "benchmarkFeedbackFlywheelStatus" in pr_comment_script
    assert "benchmarkOcrStatus" in pr_comment_script
    assert "benchmarkQdrantStatus" in pr_comment_script
    assert "assistant=${benchmarkAssistantStatus}" in pr_comment_script
    assert "review_queue=${benchmarkReviewQueueStatus}" in pr_comment_script
    assert "feedback_flywheel=${benchmarkFeedbackFlywheelStatus}" in pr_comment_script
    assert "ocr=${benchmarkOcrStatus}" in pr_comment_script
    assert "qdrant=${benchmarkQdrantStatus}" in pr_comment_script
    assert "Benchmark Feedback Flywheel" in pr_comment_script
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
