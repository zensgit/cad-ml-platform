"use strict";

function envStr(name, fallback = "") {
  const value = process.env[name];
  if (value === undefined || value === null || value === "") {
    return String(fallback);
  }
  return String(value);
}

function loadContextEnv(processLike) {
  const env = processLike.env || {};
  const inputs = JSON.parse(env.GITHUB_EVENT_INPUTS_JSON || "{}");
  const steps = JSON.parse(env.GITHUB_STEPS_JSON || "{}");
  const mapping = {"STEP_EVALUATION_COMBINED_SCORE": ["steps", "evaluation", "combined_score"], "STEP_EVALUATION_VISION_SCORE": ["steps", "evaluation", "vision_score"], "STEP_EVALUATION_OCR_SCORE": ["steps", "evaluation", "ocr_score"], "WF_INPUT_MIN_COMBINED": ["inputs", "min_combined"], "WF_INPUT_MIN_VISION": ["inputs", "min_vision"], "WF_INPUT_MIN_OCR": ["inputs", "min_ocr"], "STEP_INSIGHTS_HAS_ANOMALIES": ["steps", "insights", "has_anomalies"], "STEP_SECURITY_SECURITY_STATUS": ["steps", "security", "security_status"], "STEP_GRAPH2D_REVIEW_PACK_ENABLED": ["steps", "graph2d_review_pack", "enabled"], "STEP_GRAPH2D_REVIEW_GATE_ENABLED": ["steps", "graph2d_review_gate", "enabled"], "STEP_GRAPH2D_TRAIN_SWEEP_ENABLED": ["steps", "graph2d_train_sweep", "enabled"], "STEP_GRAPH2D_REVIEW_PACK_INPUT_CSV": ["steps", "graph2d_review_pack", "input_csv"], "STEP_GRAPH2D_REVIEW_PACK_INPUT_SOURCE": ["steps", "graph2d_review_pack", "input_source"], "STEP_GRAPH2D_REVIEW_PACK_CANDIDATE_ROWS": ["steps", "graph2d_review_pack", "candidate_rows"], "STEP_GRAPH2D_REVIEW_PACK_HYBRID_REJECTED_COUNT": ["steps", "graph2d_review_pack", "hybrid_rejected_count"], "STEP_GRAPH2D_REVIEW_PACK_CONFLICT_COUNT": ["steps", "graph2d_review_pack", "conflict_count"], "STEP_GRAPH2D_REVIEW_PACK_TOP_REVIEW_REASONS": ["steps", "graph2d_review_pack", "top_review_reasons"], "STEP_GRAPH2D_REVIEW_PACK_TOP_REVIEW_PRIORITIES": ["steps", "graph2d_review_pack", "top_review_priorities"], "STEP_GRAPH2D_REVIEW_PACK_TOP_CONFIDENCE_BANDS": ["steps", "graph2d_review_pack", "top_confidence_bands"], "STEP_GRAPH2D_REVIEW_PACK_TOP_PRIMARY_SOURCES": ["steps", "graph2d_review_pack", "top_primary_sources"], "STEP_GRAPH2D_REVIEW_PACK_TOP_SHADOW_SOURCES": ["steps", "graph2d_review_pack", "top_shadow_sources"], "STEP_GRAPH2D_REVIEW_PACK_TOP_KNOWLEDGE_CHECK_CATEGORIES": ["steps", "graph2d_review_pack", "top_knowledge_check_categories"], "STEP_GRAPH2D_REVIEW_PACK_TOP_STANDARD_CANDIDATE_TYPES": ["steps", "graph2d_review_pack", "top_standard_candidate_types"], "STEP_GRAPH2D_REVIEW_PACK_SAMPLE_EXPLANATIONS": ["steps", "graph2d_review_pack", "sample_explanations"], "STEP_GRAPH2D_REVIEW_GATE_STATUS": ["steps", "graph2d_review_gate", "status"], "STEP_GRAPH2D_REVIEW_GATE_EXIT_CODE": ["steps", "graph2d_review_gate", "exit_code"], "STEP_GRAPH2D_REVIEW_GATE_HEADLINE": ["steps", "graph2d_review_gate", "headline"], "STEP_GRAPH2D_REVIEW_GATE_STRICT_STRICT_MODE": ["steps", "graph2d_review_gate_strict", "strict_mode"], "STEP_GRAPH2D_REVIEW_GATE_STRICT_SHOULD_FAIL": ["steps", "graph2d_review_gate_strict", "should_fail"], "STEP_GRAPH2D_REVIEW_GATE_STRICT_REASON": ["steps", "graph2d_review_gate_strict", "reason"], "STEP_GRAPH2D_TRAIN_SWEEP_TOTAL_RUNS": ["steps", "graph2d_train_sweep", "total_runs"], "STEP_GRAPH2D_TRAIN_SWEEP_FAILED_RUNS": ["steps", "graph2d_train_sweep", "failed_runs"], "STEP_GRAPH2D_TRAIN_SWEEP_BEST_RECIPE": ["steps", "graph2d_train_sweep", "best_recipe"], "STEP_GRAPH2D_TRAIN_SWEEP_BEST_SEED": ["steps", "graph2d_train_sweep", "best_seed"], "STEP_GRAPH2D_TRAIN_SWEEP_RECOMMENDED_ENV_FILE": ["steps", "graph2d_train_sweep", "recommended_env_file"], "STEP_GRAPH2D_TRAIN_SWEEP_BEST_RUN_SCRIPT": ["steps", "graph2d_train_sweep", "best_run_script"], "STEP_BENCHMARK_SCORECARD_ENABLED": ["steps", "benchmark_scorecard", "enabled"], "STEP_BENCHMARK_SCORECARD_OVERALL_STATUS": ["steps", "benchmark_scorecard", "overall_status"], "STEP_BENCHMARK_SCORECARD_HYBRID_STATUS": ["steps", "benchmark_scorecard", "hybrid_status"], "STEP_BENCHMARK_SCORECARD_GRAPH2D_STATUS": ["steps", "benchmark_scorecard", "graph2d_status"], "STEP_BENCHMARK_SCORECARD_HISTORY_STATUS": ["steps", "benchmark_scorecard", "history_status"], "STEP_BENCHMARK_SCORECARD_BREP_STATUS": ["steps", "benchmark_scorecard", "brep_status"], "STEP_BENCHMARK_SCORECARD_GOVERNANCE_STATUS": ["steps", "benchmark_scorecard", "governance_status"], "STEP_BENCHMARK_SCORECARD_ASSISTANT_STATUS": ["steps", "benchmark_scorecard", "assistant_status"], "STEP_BENCHMARK_SCORECARD_REVIEW_QUEUE_STATUS": ["steps", "benchmark_scorecard", "review_queue_status"], "STEP_BENCHMARK_SCORECARD_FEEDBACK_FLYWHEEL_STATUS": ["steps", "benchmark_scorecard", "feedback_flywheel_status"], "STEP_BENCHMARK_SCORECARD_OCR_STATUS": ["steps", "benchmark_scorecard", "ocr_status"], "STEP_BENCHMARK_SCORECARD_QDRANT_STATUS": ["steps", "benchmark_scorecard", "qdrant_status"], "STEP_BENCHMARK_SCORECARD_KNOWLEDGE_STATUS": ["steps", "benchmark_scorecard", "knowledge_status"], "STEP_BENCHMARK_SCORECARD_KNOWLEDGE_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_scorecard", "knowledge_total_reference_items"], "STEP_BENCHMARK_SCORECARD_KNOWLEDGE_FOCUS_AREAS": ["steps", "benchmark_scorecard", "knowledge_focus_areas"], "STEP_BENCHMARK_SCORECARD_ENGINEERING_STATUS": ["steps", "benchmark_scorecard", "engineering_status"], "STEP_BENCHMARK_SCORECARD_ENGINEERING_COVERAGE_RATIO": ["steps", "benchmark_scorecard", "engineering_coverage_ratio"], "STEP_BENCHMARK_SCORECARD_ENGINEERING_TOP_STANDARD_TYPES": ["steps", "benchmark_scorecard", "engineering_top_standard_types"], "STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_scorecard", "operator_adoption_status"], "STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_MODE": ["steps", "benchmark_scorecard", "operator_adoption_mode"], "STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_scorecard", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_scorecard", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_scorecard", "recommendations"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_ENABLED": ["steps", "benchmark_engineering_signals", "enabled"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_STATUS": ["steps", "benchmark_engineering_signals", "status"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_COVERAGE_RATIO": ["steps", "benchmark_engineering_signals", "coverage_ratio"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_ROWS_WITH_VIOLATIONS": ["steps", "benchmark_engineering_signals", "rows_with_violations"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_ROWS_WITH_STANDARDS_CANDIDATES": ["steps", "benchmark_engineering_signals", "rows_with_standards_candidates"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_OCR_STANDARD_SIGNAL_COUNT": ["steps", "benchmark_engineering_signals", "ocr_standard_signal_count"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_RECOMMENDATIONS": ["steps", "benchmark_engineering_signals", "recommendations"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_MD": ["steps", "benchmark_engineering_signals", "output_md"], "STEP_BENCHMARK_REALDATA_SIGNALS_ENABLED": ["steps", "benchmark_realdata_signals", "enabled"], "STEP_BENCHMARK_REALDATA_SIGNALS_STATUS": ["steps", "benchmark_realdata_signals", "status"], "STEP_BENCHMARK_REALDATA_SIGNALS_READY_COMPONENT_COUNT": ["steps", "benchmark_realdata_signals", "ready_component_count"], "STEP_BENCHMARK_REALDATA_SIGNALS_PARTIAL_COMPONENT_COUNT": ["steps", "benchmark_realdata_signals", "partial_component_count"], "STEP_BENCHMARK_REALDATA_SIGNALS_ENVIRONMENT_BLOCKED_COUNT": ["steps", "benchmark_realdata_signals", "environment_blocked_count"], "STEP_BENCHMARK_REALDATA_SIGNALS_AVAILABLE_COMPONENT_COUNT": ["steps", "benchmark_realdata_signals", "available_component_count"], "STEP_BENCHMARK_REALDATA_SIGNALS_HYBRID_DXF_STATUS": ["steps", "benchmark_realdata_signals", "hybrid_dxf_status"], "STEP_BENCHMARK_REALDATA_SIGNALS_HISTORY_H5_STATUS": ["steps", "benchmark_realdata_signals", "history_h5_status"], "STEP_BENCHMARK_REALDATA_SIGNALS_STEP_SMOKE_STATUS": ["steps", "benchmark_realdata_signals", "step_smoke_status"], "STEP_BENCHMARK_REALDATA_SIGNALS_STEP_DIR_STATUS": ["steps", "benchmark_realdata_signals", "step_dir_status"], "STEP_BENCHMARK_REALDATA_SIGNALS_RECOMMENDATIONS": ["steps", "benchmark_realdata_signals", "recommendations"], "STEP_BENCHMARK_REALDATA_SIGNALS_OUTPUT_MD": ["steps", "benchmark_realdata_signals", "output_md"], "STEP_BENCHMARK_REALDATA_SCORECARD_ENABLED": ["steps", "benchmark_realdata_scorecard", "enabled"], "STEP_BENCHMARK_REALDATA_SCORECARD_STATUS": ["steps", "benchmark_realdata_scorecard", "status"], "STEP_BENCHMARK_REALDATA_SCORECARD_READY_COMPONENT_COUNT": ["steps", "benchmark_realdata_scorecard", "ready_component_count"], "STEP_BENCHMARK_REALDATA_SCORECARD_PARTIAL_COMPONENT_COUNT": ["steps", "benchmark_realdata_scorecard", "partial_component_count"], "STEP_BENCHMARK_REALDATA_SCORECARD_ENVIRONMENT_BLOCKED_COUNT": ["steps", "benchmark_realdata_scorecard", "environment_blocked_count"], "STEP_BENCHMARK_REALDATA_SCORECARD_AVAILABLE_COMPONENT_COUNT": ["steps", "benchmark_realdata_scorecard", "available_component_count"], "STEP_BENCHMARK_REALDATA_SCORECARD_BEST_SURFACE": ["steps", "benchmark_realdata_scorecard", "best_surface"], "STEP_BENCHMARK_REALDATA_SCORECARD_HYBRID_DXF_STATUS": ["steps", "benchmark_realdata_scorecard", "hybrid_dxf_status"], "STEP_BENCHMARK_REALDATA_SCORECARD_HISTORY_H5_STATUS": ["steps", "benchmark_realdata_scorecard", "history_h5_status"], "STEP_BENCHMARK_REALDATA_SCORECARD_STEP_SMOKE_STATUS": ["steps", "benchmark_realdata_scorecard", "step_smoke_status"], "STEP_BENCHMARK_REALDATA_SCORECARD_STEP_DIR_STATUS": ["steps", "benchmark_realdata_scorecard", "step_dir_status"], "STEP_BENCHMARK_REALDATA_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_realdata_scorecard", "recommendations"], "STEP_BENCHMARK_REALDATA_SCORECARD_OUTPUT_MD": ["steps", "benchmark_realdata_scorecard", "output_md"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_ENABLED": ["steps", "benchmark_competitive_surpass_index", "enabled"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_STATUS": ["steps", "benchmark_competitive_surpass_index", "status"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_SCORE": ["steps", "benchmark_competitive_surpass_index", "score"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_READY_PILLARS": ["steps", "benchmark_competitive_surpass_index", "ready_pillars"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_PARTIAL_PILLARS": ["steps", "benchmark_competitive_surpass_index", "partial_pillars"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_BLOCKED_PILLARS": ["steps", "benchmark_competitive_surpass_index", "blocked_pillars"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_PRIMARY_GAPS": ["steps", "benchmark_competitive_surpass_index", "primary_gaps"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_RECOMMENDATIONS": ["steps", "benchmark_competitive_surpass_index", "recommendations"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_MD": ["steps", "benchmark_competitive_surpass_index", "output_md"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_ENABLED": ["steps", "benchmark_competitive_surpass_trend", "enabled"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_STATUS": ["steps", "benchmark_competitive_surpass_trend", "status"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_SCORE_DELTA": ["steps", "benchmark_competitive_surpass_trend", "score_delta"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_PILLAR_IMPROVEMENTS": ["steps", "benchmark_competitive_surpass_trend", "pillar_improvements"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_PILLAR_REGRESSIONS": ["steps", "benchmark_competitive_surpass_trend", "pillar_regressions"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_RESOLVED_PRIMARY_GAPS": ["steps", "benchmark_competitive_surpass_trend", "resolved_primary_gaps"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_NEW_PRIMARY_GAPS": ["steps", "benchmark_competitive_surpass_trend", "new_primary_gaps"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS": ["steps", "benchmark_competitive_surpass_trend", "recommendations"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_OUTPUT_MD": ["steps", "benchmark_competitive_surpass_trend", "output_md"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_ENABLED": ["steps", "benchmark_competitive_surpass_action_plan", "enabled"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS": ["steps", "benchmark_competitive_surpass_action_plan", "status"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_competitive_surpass_action_plan", "total_action_count"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_competitive_surpass_action_plan", "high_priority_action_count"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT": ["steps", "benchmark_competitive_surpass_action_plan", "medium_priority_action_count"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS": ["steps", "benchmark_competitive_surpass_action_plan", "priority_pillars"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_competitive_surpass_action_plan", "recommended_first_actions"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_competitive_surpass_action_plan", "recommendations"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_OUTPUT_MD": ["steps", "benchmark_competitive_surpass_action_plan", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_ENABLED": ["steps", "benchmark_knowledge_readiness", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_STATUS": ["steps", "benchmark_knowledge_readiness", "status"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_knowledge_readiness", "total_reference_items"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_READY_COMPONENT_COUNT": ["steps", "benchmark_knowledge_readiness", "ready_component_count"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_PARTIAL_COMPONENT_COUNT": ["steps", "benchmark_knowledge_readiness", "partial_component_count"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_MISSING_COMPONENT_COUNT": ["steps", "benchmark_knowledge_readiness", "missing_component_count"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_FOCUS_AREA_COUNT": ["steps", "benchmark_knowledge_readiness", "focus_area_count"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_FOCUS_AREAS": ["steps", "benchmark_knowledge_readiness", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_DOMAIN_COUNT": ["steps", "benchmark_knowledge_readiness", "domain_count"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_readiness", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_DOMAIN_FOCUS_AREAS": ["steps", "benchmark_knowledge_readiness", "domain_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_RECOMMENDATIONS": ["steps", "benchmark_knowledge_readiness", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_MD": ["steps", "benchmark_knowledge_readiness", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_ENABLED": ["steps", "benchmark_knowledge_drift", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_knowledge_drift", "status"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_CURRENT_STATUS": ["steps", "benchmark_knowledge_drift", "current_status"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_PREVIOUS_STATUS": ["steps", "benchmark_knowledge_drift", "previous_status"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_REFERENCE_ITEM_DELTA": ["steps", "benchmark_knowledge_drift", "reference_item_delta"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_REGRESSIONS": ["steps", "benchmark_knowledge_drift", "regressions"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_IMPROVEMENTS": ["steps", "benchmark_knowledge_drift", "improvements"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_RESOLVED_FOCUS_AREAS": ["steps", "benchmark_knowledge_drift", "resolved_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_NEW_FOCUS_AREAS": ["steps", "benchmark_knowledge_drift", "new_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_knowledge_drift", "domain_regressions"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_knowledge_drift", "domain_improvements"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_drift", "resolved_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_drift", "new_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_knowledge_drift", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_MD": ["steps", "benchmark_knowledge_drift", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_ENABLED": ["steps", "benchmark_knowledge_application", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_STATUS": ["steps", "benchmark_knowledge_application", "status"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_application", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_application", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_MISSING_DOMAIN_COUNT": ["steps", "benchmark_knowledge_application", "missing_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_application", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_FOCUS_AREA_COUNT": ["steps", "benchmark_knowledge_application", "focus_area_count"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_FOCUS_AREAS": ["steps", "benchmark_knowledge_application", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_application", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_application", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_RECOMMENDATIONS": ["steps", "benchmark_knowledge_application", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_MD": ["steps", "benchmark_knowledge_application", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_ENABLED": ["steps", "benchmark_knowledge_realdata_correlation", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_STATUS": ["steps", "benchmark_knowledge_realdata_correlation", "status"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_realdata_correlation", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_realdata_correlation", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_realdata_correlation", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_realdata_correlation", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_knowledge_realdata_correlation", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_realdata_correlation", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_realdata_correlation", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_knowledge_realdata_correlation", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_OUTPUT_MD": ["steps", "benchmark_knowledge_realdata_correlation", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_ENABLED": ["steps", "benchmark_knowledge_domain_matrix", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_STATUS": ["steps", "benchmark_knowledge_domain_matrix", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_matrix", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_matrix", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_matrix", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_matrix", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS": ["steps", "benchmark_knowledge_domain_matrix", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_matrix", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_domain_matrix", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_matrix", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_matrix", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_ENABLED": ["steps", "benchmark_knowledge_domain_capability_matrix", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS": ["steps", "benchmark_knowledge_domain_capability_matrix", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_capability_matrix", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_capability_matrix", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_capability_matrix", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_capability_matrix", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS": ["steps", "benchmark_knowledge_domain_capability_matrix", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_capability_matrix", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PROVIDER_GAP_DOMAINS": ["steps", "benchmark_knowledge_domain_capability_matrix", "provider_gap_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_SURFACE_GAP_DOMAINS": ["steps", "benchmark_knowledge_domain_capability_matrix", "surface_gap_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_domain_capability_matrix", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_capability_matrix", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_capability_matrix", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_ENABLED": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_STATUS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_TOTAL_API_ROUTE_COUNT": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "total_api_route_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_FOCUS_AREAS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PUBLIC_API_GAP_DOMAINS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "public_api_gap_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_REFERENCE_GAP_DOMAINS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "reference_gap_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_ENABLED": ["steps", "benchmark_knowledge_domain_capability_drift", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS": ["steps", "benchmark_knowledge_domain_capability_drift", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_CURRENT_STATUS": ["steps", "benchmark_knowledge_domain_capability_drift", "current_status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_PREVIOUS_STATUS": ["steps", "benchmark_knowledge_domain_capability_drift", "previous_status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_PROVIDER_GAP_DELTA": ["steps", "benchmark_knowledge_domain_capability_drift", "provider_gap_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_SURFACE_GAP_DELTA": ["steps", "benchmark_knowledge_domain_capability_drift", "surface_gap_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_knowledge_domain_capability_drift", "domain_regressions"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_knowledge_domain_capability_drift", "domain_improvements"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_capability_drift", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_capability_drift", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ENABLED": ["steps", "benchmark_knowledge_domain_action_plan", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS": ["steps", "benchmark_knowledge_domain_action_plan", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "total_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "high_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_action_plan", "medium_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_action_plan", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_knowledge_domain_action_plan", "recommended_first_actions"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_DOMAIN_ACTION_COUNTS": ["steps", "benchmark_knowledge_domain_action_plan", "domain_action_counts"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_action_plan", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_action_plan", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_ENABLED": ["steps", "benchmark_knowledge_domain_surface_action_plan", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_STATUS": ["steps", "benchmark_knowledge_domain_surface_action_plan", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_TOTAL_SUBCAPABILITY_COUNT": ["steps", "benchmark_knowledge_domain_surface_action_plan", "total_subcapability_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_surface_action_plan", "total_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_surface_action_plan", "high_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_surface_action_plan", "medium_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_surface_action_plan", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_knowledge_domain_surface_action_plan", "recommended_first_actions"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_DOMAIN_ACTION_COUNTS": ["steps", "benchmark_knowledge_domain_surface_action_plan", "domain_action_counts"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_surface_action_plan", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_surface_action_plan", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_ENABLED": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "total_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "high_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "medium_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "gate_open"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "recommended_first_actions"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_release_readiness_action_plan", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_ENABLED": ["steps", "benchmark_knowledge_domain_control_plane", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS": ["steps", "benchmark_knowledge_domain_control_plane", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_MISSING_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "missing_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_TOTAL_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "total_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_domain_control_plane", "high_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS": ["steps", "benchmark_knowledge_domain_control_plane", "release_blockers"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_control_plane", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_FOCUS_AREAS": ["steps", "benchmark_knowledge_domain_control_plane", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_control_plane", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_control_plane", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_ENABLED": ["steps", "benchmark_knowledge_domain_control_plane_drift", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_CURRENT_STATUS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "current_status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_PREVIOUS_STATUS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "previous_status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_READY_DOMAIN_DELTA": ["steps", "benchmark_knowledge_domain_control_plane_drift", "ready_domain_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_BLOCKED_DOMAIN_DELTA": ["steps", "benchmark_knowledge_domain_control_plane_drift", "blocked_domain_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_TOTAL_ACTION_DELTA": ["steps", "benchmark_knowledge_domain_control_plane_drift", "total_action_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_HIGH_PRIORITY_ACTION_DELTA": ["steps", "benchmark_knowledge_domain_control_plane_drift", "high_priority_action_delta"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "domain_regressions"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "domain_improvements"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "resolved_release_blockers"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "new_release_blockers"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_control_plane_drift", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_control_plane_drift", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_ENABLED": ["steps", "benchmark_knowledge_domain_release_gate", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS": ["steps", "benchmark_knowledge_domain_release_gate", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN": ["steps", "benchmark_knowledge_domain_release_gate", "gate_open"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_release_gate", "releasable_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_release_gate", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_domain_release_gate", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS": ["steps", "benchmark_knowledge_domain_release_gate", "blocking_reasons"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_WARNING_REASONS": ["steps", "benchmark_knowledge_domain_release_gate", "warning_reasons"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS": ["steps", "benchmark_knowledge_domain_release_gate", "releasable_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS": ["steps", "benchmark_knowledge_domain_release_gate", "blocked_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_domain_release_gate", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDED_FIRST_ACTION": ["steps", "benchmark_knowledge_domain_release_gate", "recommended_first_action"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS": ["steps", "benchmark_knowledge_domain_release_gate", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_release_gate", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_ENABLED": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "status"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "summary"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "mismatches"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_OUTPUT_MD": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_ENABLED": ["steps", "benchmark_knowledge_reference_inventory", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_STATUS": ["steps", "benchmark_knowledge_reference_inventory", "status"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_knowledge_reference_inventory", "total_reference_items"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_TABLE_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "total_table_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_POPULATED_TABLE_COUNT": ["steps", "benchmark_knowledge_reference_inventory", "populated_table_count"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_reference_inventory", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_FOCUS_TABLES": ["steps", "benchmark_knowledge_reference_inventory", "focus_tables"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS": ["steps", "benchmark_knowledge_reference_inventory", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_MD": ["steps", "benchmark_knowledge_reference_inventory", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_ENABLED": ["steps", "benchmark_knowledge_source_coverage", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_STATUS": ["steps", "benchmark_knowledge_source_coverage", "status"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_READY_SOURCE_GROUP_COUNT": ["steps", "benchmark_knowledge_source_coverage", "ready_source_group_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_PARTIAL_SOURCE_GROUP_COUNT": ["steps", "benchmark_knowledge_source_coverage", "partial_source_group_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_MISSING_SOURCE_GROUP_COUNT": ["steps", "benchmark_knowledge_source_coverage", "missing_source_group_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_GROUP_COUNT": ["steps", "benchmark_knowledge_source_coverage", "total_source_group_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_TABLE_COUNT": ["steps", "benchmark_knowledge_source_coverage", "total_source_table_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_ITEM_COUNT": ["steps", "benchmark_knowledge_source_coverage", "total_source_item_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_REFERENCE_STANDARD_COUNT": ["steps", "benchmark_knowledge_source_coverage", "total_reference_standard_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_READY_EXPANSION_CANDIDATE_COUNT": ["steps", "benchmark_knowledge_source_coverage", "ready_expansion_candidate_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_FOCUS_AREAS": ["steps", "benchmark_knowledge_source_coverage", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_source_coverage", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_source_coverage", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES": ["steps", "benchmark_knowledge_source_coverage", "expansion_candidates"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS": ["steps", "benchmark_knowledge_source_coverage", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_OUTPUT_MD": ["steps", "benchmark_knowledge_source_coverage", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_ENABLED": ["steps", "benchmark_knowledge_source_action_plan", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS": ["steps", "benchmark_knowledge_source_action_plan", "status"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_knowledge_source_action_plan", "total_action_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_source_action_plan", "high_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT": ["steps", "benchmark_knowledge_source_action_plan", "medium_priority_action_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_EXPANSION_ACTION_COUNT": ["steps", "benchmark_knowledge_source_action_plan", "expansion_action_count"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_source_action_plan", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_knowledge_source_action_plan", "recommended_first_actions"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS": ["steps", "benchmark_knowledge_source_action_plan", "source_group_action_counts"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_knowledge_source_action_plan", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_MD": ["steps", "benchmark_knowledge_source_action_plan", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_ENABLED": ["steps", "benchmark_knowledge_source_drift", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_STATUS": ["steps", "benchmark_knowledge_source_drift", "status"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_CURRENT_STATUS": ["steps", "benchmark_knowledge_source_drift", "current_status"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_PREVIOUS_STATUS": ["steps", "benchmark_knowledge_source_drift", "previous_status"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_READY_SOURCE_GROUP_DELTA": ["steps", "benchmark_knowledge_source_drift", "ready_source_group_delta"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_MISSING_SOURCE_GROUP_DELTA": ["steps", "benchmark_knowledge_source_drift", "missing_source_group_delta"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_REGRESSIONS": ["steps", "benchmark_knowledge_source_drift", "regressions"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_IMPROVEMENTS": ["steps", "benchmark_knowledge_source_drift", "improvements"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS": ["steps", "benchmark_knowledge_source_drift", "source_group_regressions"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS": ["steps", "benchmark_knowledge_source_drift", "source_group_improvements"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_FOCUS_AREAS": ["steps", "benchmark_knowledge_source_drift", "resolved_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_NEW_FOCUS_AREAS": ["steps", "benchmark_knowledge_source_drift", "new_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_source_drift", "resolved_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_source_drift", "new_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_knowledge_source_drift", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_OUTPUT_MD": ["steps", "benchmark_knowledge_source_drift", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_ENABLED": ["steps", "benchmark_knowledge_outcome_correlation", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_STATUS": ["steps", "benchmark_knowledge_outcome_correlation", "status"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_READY_DOMAIN_COUNT": ["steps", "benchmark_knowledge_outcome_correlation", "ready_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_PARTIAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_outcome_correlation", "partial_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_BLOCKED_DOMAIN_COUNT": ["steps", "benchmark_knowledge_outcome_correlation", "blocked_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_TOTAL_DOMAIN_COUNT": ["steps", "benchmark_knowledge_outcome_correlation", "total_domain_count"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_knowledge_outcome_correlation", "focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_outcome_correlation", "priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_knowledge_outcome_correlation", "domain_statuses"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_knowledge_outcome_correlation", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_OUTPUT_MD": ["steps", "benchmark_knowledge_outcome_correlation", "output_md"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_ENABLED": ["steps", "benchmark_knowledge_outcome_drift", "enabled"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_knowledge_outcome_drift", "status"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_CURRENT_STATUS": ["steps", "benchmark_knowledge_outcome_drift", "current_status"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_PREVIOUS_STATUS": ["steps", "benchmark_knowledge_outcome_drift", "previous_status"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_READY_DOMAIN_DELTA": ["steps", "benchmark_knowledge_outcome_drift", "ready_domain_delta"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_BLOCKED_DOMAIN_DELTA": ["steps", "benchmark_knowledge_outcome_drift", "blocked_domain_delta"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_REGRESSIONS": ["steps", "benchmark_knowledge_outcome_drift", "regressions"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_IMPROVEMENTS": ["steps", "benchmark_knowledge_outcome_drift", "improvements"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_knowledge_outcome_drift", "domain_regressions"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_knowledge_outcome_drift", "domain_improvements"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_FOCUS_AREAS": ["steps", "benchmark_knowledge_outcome_drift", "resolved_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_NEW_FOCUS_AREAS": ["steps", "benchmark_knowledge_outcome_drift", "new_focus_areas"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_outcome_drift", "resolved_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_knowledge_outcome_drift", "new_priority_domains"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_knowledge_outcome_drift", "recommendations"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_OUTPUT_MD": ["steps", "benchmark_knowledge_outcome_drift", "output_md"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_ENABLED": ["steps", "feedback_flywheel_benchmark", "enabled"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_STATUS": ["steps", "feedback_flywheel_benchmark", "status"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_FEEDBACK_TOTAL": ["steps", "feedback_flywheel_benchmark", "feedback_total"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_CORRECTION_COUNT": ["steps", "feedback_flywheel_benchmark", "correction_count"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_FINETUNE_SAMPLE_COUNT": ["steps", "feedback_flywheel_benchmark", "finetune_sample_count"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_METRIC_TRIPLET_COUNT": ["steps", "feedback_flywheel_benchmark", "metric_triplet_count"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_MD": ["steps", "feedback_flywheel_benchmark", "output_md"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_ENABLED": ["steps", "benchmark_operational_summary", "enabled"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OVERALL_STATUS": ["steps", "benchmark_operational_summary", "overall_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_FEEDBACK_STATUS": ["steps", "benchmark_operational_summary", "feedback_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_ASSISTANT_STATUS": ["steps", "benchmark_operational_summary", "assistant_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_REVIEW_QUEUE_STATUS": ["steps", "benchmark_operational_summary", "review_queue_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OCR_STATUS": ["steps", "benchmark_operational_summary", "ocr_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_operational_summary", "operator_adoption_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_operational_summary", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_operational_summary", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_BLOCKERS": ["steps", "benchmark_operational_summary", "blockers"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_RECOMMENDATIONS": ["steps", "benchmark_operational_summary", "recommendations"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_MD": ["steps", "benchmark_operational_summary", "output_md"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_ENABLED": ["steps", "benchmark_artifact_bundle", "enabled"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OVERALL_STATUS": ["steps", "benchmark_artifact_bundle", "overall_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_AVAILABLE_ARTIFACT_COUNT": ["steps", "benchmark_artifact_bundle", "available_artifact_count"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_STATUS": ["steps", "benchmark_artifact_bundle", "feedback_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_STATUS": ["steps", "benchmark_artifact_bundle", "assistant_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_STATUS": ["steps", "benchmark_artifact_bundle", "review_queue_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OCR_STATUS": ["steps", "benchmark_artifact_bundle", "ocr_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "knowledge_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_COMPONENT_CHANGES": ["steps", "benchmark_artifact_bundle", "knowledge_drift_component_changes"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_domain_regressions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_domain_improvements"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_resolved_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_drift_new_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_application_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_application_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_application_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_application_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_application_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_realdata_correlation_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_realdata_correlation_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_realdata_correlation_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_realdata_correlation_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_realdata_correlation_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_matrix_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_matrix_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_matrix_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_domain_matrix_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_matrix_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_matrix_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_matrix_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_matrix_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_matrix_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_matrix_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_action_plan_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_action_plan_actions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_action_plan_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_action_plan_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_total_action_count"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_gate_open"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_recommended_first_actions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_readiness_action_plan_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_release_blockers"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_domain_regressions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_domain_improvements"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_resolved_release_blockers"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_new_release_blockers"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_control_plane_drift_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_SUMMARY": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_gate_open"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_blocking_reasons"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_releasable_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_blocked_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_gate_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_reference_inventory_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_reference_inventory_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_artifact_bundle", "knowledge_reference_inventory_total_reference_items"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_source_coverage_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_source_coverage_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES": ["steps", "benchmark_artifact_bundle", "knowledge_source_coverage_expansion_candidates"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_source_coverage_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_source_action_plan_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_source_action_plan_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_artifact_bundle", "knowledge_source_action_plan_recommended_first_actions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS": ["steps", "benchmark_artifact_bundle", "knowledge_source_action_plan_source_group_action_counts"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_source_action_plan_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_source_group_regressions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_source_group_improvements"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_resolved_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_new_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_source_drift_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_correlation_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_correlation_focus_areas"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_correlation_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_correlation_domain_statuses"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_correlation_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_domain_regressions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_domain_improvements"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_resolved_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_new_priority_domains"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_outcome_drift_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_STATUS": ["steps", "benchmark_artifact_bundle", "engineering_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_STATUS": ["steps", "benchmark_artifact_bundle", "realdata_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_STATUS": ["steps", "benchmark_artifact_bundle", "realdata_scorecard_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "realdata_scorecard_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "realdata_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "operator_adoption_knowledge_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "operator_adoption_knowledge_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_artifact_bundle", "scorecard_operator_adoption_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_MODE": ["steps", "benchmark_artifact_bundle", "scorecard_operator_adoption_mode"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "scorecard_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "scorecard_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_artifact_bundle", "operational_operator_adoption_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "operational_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_artifact_bundle", "operational_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_artifact_bundle", "operator_adoption_release_surface_alignment_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_artifact_bundle", "operator_adoption_release_surface_alignment_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_artifact_bundle", "operator_adoption_release_surface_alignment_mismatches"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_surface_alignment_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_surface_alignment_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_artifact_bundle", "knowledge_domain_release_surface_alignment_mismatches"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_STATUS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_index_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_PRIMARY_GAPS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_primary_gaps"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_STATUS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_trend_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_SUMMARY": ["steps", "benchmark_artifact_bundle", "competitive_surpass_trend_summary"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_trend_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_action_plan_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_artifact_bundle", "competitive_surpass_action_plan_total_action_count"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_action_plan_priority_pillars"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "competitive_surpass_action_plan_recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_BLOCKERS": ["steps", "benchmark_artifact_bundle", "blockers"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "recommendations"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_MD": ["steps", "benchmark_artifact_bundle", "output_md"], "STEP_BENCHMARK_COMPANION_SUMMARY_ENABLED": ["steps", "benchmark_companion_summary", "enabled"], "STEP_BENCHMARK_COMPANION_SUMMARY_OVERALL_STATUS": ["steps", "benchmark_companion_summary", "overall_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_REVIEW_SURFACE": ["steps", "benchmark_companion_summary", "review_surface"], "STEP_BENCHMARK_COMPANION_SUMMARY_PRIMARY_GAP": ["steps", "benchmark_companion_summary", "primary_gap"], "STEP_BENCHMARK_COMPANION_SUMMARY_HYBRID_STATUS": ["steps", "benchmark_companion_summary", "hybrid_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_ASSISTANT_STATUS": ["steps", "benchmark_companion_summary", "assistant_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_REVIEW_QUEUE_STATUS": ["steps", "benchmark_companion_summary", "review_queue_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OCR_STATUS": ["steps", "benchmark_companion_summary", "ocr_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_QDRANT_STATUS": ["steps", "benchmark_companion_summary", "qdrant_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_STATUS": ["steps", "benchmark_companion_summary", "knowledge_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "knowledge_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_drift_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_COMPONENT_CHANGES": ["steps", "benchmark_companion_summary", "knowledge_drift_component_changes"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_companion_summary", "knowledge_drift_domain_regressions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_companion_summary", "knowledge_drift_domain_improvements"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_drift_resolved_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_drift_new_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_domain_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_STATUS": ["steps", "benchmark_companion_summary", "knowledge_application_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_application_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_application_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_application_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_application_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_STATUS": ["steps", "benchmark_companion_summary", "knowledge_realdata_correlation_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_realdata_correlation_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_realdata_correlation_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_realdata_correlation_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_realdata_correlation_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_matrix_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_domain_matrix_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_matrix_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_domain_matrix_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_matrix_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_matrix_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_matrix_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_matrix_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_matrix_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_matrix_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_action_plan_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_action_plan_actions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_action_plan_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_action_plan_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_total_action_count"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_gate_open"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_recommended_first_actions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_readiness_action_plan_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_release_blockers"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_domain_regressions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_domain_improvements"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_resolved_release_blockers"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_new_release_blockers"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_control_plane_drift_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_SUMMARY": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_gate_open"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_blocking_reasons"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_releasable_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_blocked_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_gate_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_STATUS": ["steps", "benchmark_companion_summary", "knowledge_reference_inventory_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_reference_inventory_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_companion_summary", "knowledge_reference_inventory_total_reference_items"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_STATUS": ["steps", "benchmark_companion_summary", "knowledge_source_coverage_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_source_coverage_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES": ["steps", "benchmark_companion_summary", "knowledge_source_coverage_expansion_candidates"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_source_coverage_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS": ["steps", "benchmark_companion_summary", "knowledge_source_action_plan_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_source_action_plan_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_companion_summary", "knowledge_source_action_plan_recommended_first_actions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS": ["steps", "benchmark_companion_summary", "knowledge_source_action_plan_source_group_action_counts"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_source_action_plan_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "knowledge_source_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_source_group_regressions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_source_group_improvements"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_resolved_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_new_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_source_drift_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_STATUS": ["steps", "benchmark_companion_summary", "knowledge_outcome_correlation_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_companion_summary", "knowledge_outcome_correlation_focus_areas"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_outcome_correlation_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_companion_summary", "knowledge_outcome_correlation_domain_statuses"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_outcome_correlation_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_domain_regressions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_domain_improvements"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_resolved_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_new_priority_domains"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_outcome_drift_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_ENGINEERING_STATUS": ["steps", "benchmark_companion_summary", "engineering_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_STATUS": ["steps", "benchmark_companion_summary", "realdata_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_SCORECARD_STATUS": ["steps", "benchmark_companion_summary", "realdata_scorecard_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "realdata_scorecard_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "realdata_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "operator_adoption_knowledge_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "operator_adoption_knowledge_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_companion_summary", "scorecard_operator_adoption_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_MODE": ["steps", "benchmark_companion_summary", "scorecard_operator_adoption_mode"], "STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "scorecard_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "scorecard_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_companion_summary", "operational_operator_adoption_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "operational_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_companion_summary", "operational_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_companion_summary", "operator_adoption_release_surface_alignment_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_companion_summary", "operator_adoption_release_surface_alignment_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_companion_summary", "operator_adoption_release_surface_alignment_mismatches"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_release_surface_alignment_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_companion_summary", "knowledge_domain_release_surface_alignment_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_companion_summary", "knowledge_domain_release_surface_alignment_mismatches"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_INDEX_STATUS": ["steps", "benchmark_companion_summary", "competitive_surpass_index_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_PRIMARY_GAPS": ["steps", "benchmark_companion_summary", "competitive_surpass_primary_gaps"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "competitive_surpass_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_STATUS": ["steps", "benchmark_companion_summary", "competitive_surpass_trend_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_SUMMARY": ["steps", "benchmark_companion_summary", "competitive_surpass_trend_summary"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "competitive_surpass_trend_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS": ["steps", "benchmark_companion_summary", "competitive_surpass_action_plan_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_companion_summary", "competitive_surpass_action_plan_total_action_count"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS": ["steps", "benchmark_companion_summary", "competitive_surpass_action_plan_priority_pillars"], "STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "competitive_surpass_action_plan_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_BLOCKERS": ["steps", "benchmark_companion_summary", "blockers"], "STEP_BENCHMARK_COMPANION_SUMMARY_RECOMMENDED_ACTIONS": ["steps", "benchmark_companion_summary", "recommended_actions"], "STEP_BENCHMARK_COMPANION_SUMMARY_OUTPUT_MD": ["steps", "benchmark_companion_summary", "output_md"], "STEP_BENCHMARK_RELEASE_DECISION_ENABLED": ["steps", "benchmark_release_decision", "enabled"], "STEP_BENCHMARK_RELEASE_DECISION_RELEASE_STATUS": ["steps", "benchmark_release_decision", "release_status"], "STEP_BENCHMARK_RELEASE_DECISION_AUTOMATION_READY": ["steps", "benchmark_release_decision", "automation_ready"], "STEP_BENCHMARK_RELEASE_DECISION_PRIMARY_SIGNAL_SOURCE": ["steps", "benchmark_release_decision", "primary_signal_source"], "STEP_BENCHMARK_RELEASE_DECISION_BLOCKING_SIGNALS": ["steps", "benchmark_release_decision", "blocking_signals"], "STEP_BENCHMARK_RELEASE_DECISION_REVIEW_SIGNALS": ["steps", "benchmark_release_decision", "review_signals"], "STEP_BENCHMARK_RELEASE_DECISION_HYBRID_STATUS": ["steps", "benchmark_release_decision", "hybrid_status"], "STEP_BENCHMARK_RELEASE_DECISION_ASSISTANT_STATUS": ["steps", "benchmark_release_decision", "assistant_status"], "STEP_BENCHMARK_RELEASE_DECISION_REVIEW_QUEUE_STATUS": ["steps", "benchmark_release_decision", "review_queue_status"], "STEP_BENCHMARK_RELEASE_DECISION_OCR_STATUS": ["steps", "benchmark_release_decision", "ocr_status"], "STEP_BENCHMARK_RELEASE_DECISION_QDRANT_STATUS": ["steps", "benchmark_release_decision", "qdrant_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_STATUS": ["steps", "benchmark_release_decision", "knowledge_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_release_decision", "knowledge_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "knowledge_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_decision", "knowledge_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_decision", "knowledge_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_domain_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_STATUS": ["steps", "benchmark_release_decision", "knowledge_application_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_application_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_application_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_application_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_application_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_STATUS": ["steps", "benchmark_release_decision", "knowledge_realdata_correlation_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_realdata_correlation_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_realdata_correlation_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_realdata_correlation_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_realdata_correlation_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_matrix_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_domain_matrix_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_matrix_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_domain_matrix_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_matrix_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_matrix_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_matrix_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_matrix_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_domain_capability_matrix_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_matrix_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_action_plan_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS": ["steps", "benchmark_release_decision", "knowledge_domain_action_plan_actions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_total_action_count"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_gate_open"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_recommended_first_actions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_release_readiness_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_release_blockers"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_resolved_release_blockers"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_new_release_blockers"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_control_plane_drift_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_STATUS": ["steps", "benchmark_release_decision", "knowledge_source_coverage_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_source_coverage_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES": ["steps", "benchmark_release_decision", "knowledge_source_coverage_expansion_candidates"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_source_coverage_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS": ["steps", "benchmark_release_decision", "knowledge_source_action_plan_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_source_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_release_decision", "knowledge_source_action_plan_recommended_first_actions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS": ["steps", "benchmark_release_decision", "knowledge_source_action_plan_source_group_action_counts"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_source_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_STATUS": ["steps", "benchmark_release_decision", "knowledge_source_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "knowledge_source_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS": ["steps", "benchmark_release_decision", "knowledge_source_drift_source_group_regressions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS": ["steps", "benchmark_release_decision", "knowledge_source_drift_source_group_improvements"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_source_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_source_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_source_drift_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_STATUS": ["steps", "benchmark_release_decision", "knowledge_outcome_correlation_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_release_decision", "knowledge_outcome_correlation_focus_areas"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_outcome_correlation_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_release_decision", "knowledge_outcome_correlation_domain_statuses"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_outcome_correlation_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_outcome_drift_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_INDEX_STATUS": ["steps", "benchmark_release_decision", "competitive_surpass_index_status"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_PRIMARY_GAPS": ["steps", "benchmark_release_decision", "competitive_surpass_primary_gaps"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "competitive_surpass_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_STATUS": ["steps", "benchmark_release_decision", "competitive_surpass_trend_status"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_SUMMARY": ["steps", "benchmark_release_decision", "competitive_surpass_trend_summary"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "competitive_surpass_trend_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS": ["steps", "benchmark_release_decision", "competitive_surpass_action_plan_status"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_release_decision", "competitive_surpass_action_plan_total_action_count"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS": ["steps", "benchmark_release_decision", "competitive_surpass_action_plan_priority_pillars"], "STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "competitive_surpass_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_ENGINEERING_STATUS": ["steps", "benchmark_release_decision", "engineering_status"], "STEP_BENCHMARK_RELEASE_DECISION_REALDATA_STATUS": ["steps", "benchmark_release_decision", "realdata_status"], "STEP_BENCHMARK_RELEASE_DECISION_REALDATA_SCORECARD_STATUS": ["steps", "benchmark_release_decision", "realdata_scorecard_status"], "STEP_BENCHMARK_RELEASE_DECISION_REALDATA_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "realdata_scorecard_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_REALDATA_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "realdata_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_decision", "operator_adoption_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_release_decision", "operator_adoption_knowledge_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "operator_adoption_knowledge_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_decision", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_decision", "scorecard_operator_adoption_status"], "STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_MODE": ["steps", "benchmark_release_decision", "scorecard_operator_adoption_mode"], "STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_decision", "scorecard_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "scorecard_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_decision", "operational_operator_adoption_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_decision", "operational_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_decision", "operational_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_release_decision", "operator_adoption_release_surface_alignment_status"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_release_decision", "operator_adoption_release_surface_alignment_summary"], "STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_release_decision", "operator_adoption_release_surface_alignment_mismatches"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_gate_open"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_releasable_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_blocked_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_blocking_reasons"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_release_gate_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_STATUS": ["steps", "benchmark_release_decision", "knowledge_reference_inventory_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_SUMMARY": ["steps", "benchmark_release_decision", "knowledge_reference_inventory_summary"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS": ["steps", "benchmark_release_decision", "knowledge_reference_inventory_priority_domains"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_release_decision", "knowledge_reference_inventory_total_reference_items"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_reference_inventory_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_OUTPUT_MD": ["steps", "benchmark_release_decision", "output_md"], "STEP_BENCHMARK_RELEASE_RUNBOOK_ENABLED": ["steps", "benchmark_release_runbook", "enabled"], "STEP_BENCHMARK_RELEASE_RUNBOOK_RELEASE_STATUS": ["steps", "benchmark_release_runbook", "release_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_READY_TO_FREEZE_BASELINE": ["steps", "benchmark_release_runbook", "ready_to_freeze_baseline"], "STEP_BENCHMARK_RELEASE_RUNBOOK_PRIMARY_SIGNAL_SOURCE": ["steps", "benchmark_release_runbook", "primary_signal_source"], "STEP_BENCHMARK_RELEASE_RUNBOOK_NEXT_ACTION": ["steps", "benchmark_release_runbook", "next_action"], "STEP_BENCHMARK_RELEASE_RUNBOOK_MISSING_ARTIFACTS": ["steps", "benchmark_release_runbook", "missing_artifacts"], "STEP_BENCHMARK_RELEASE_RUNBOOK_BLOCKING_SIGNALS": ["steps", "benchmark_release_runbook", "blocking_signals"], "STEP_BENCHMARK_RELEASE_RUNBOOK_REVIEW_SIGNALS": ["steps", "benchmark_release_runbook", "review_signals"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_STATUS": ["steps", "benchmark_release_runbook", "knowledge_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "knowledge_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "knowledge_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_runbook", "knowledge_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_runbook", "knowledge_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_domain_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_STATUS": ["steps", "benchmark_release_runbook", "knowledge_application_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_application_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_application_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_application_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_application_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_STATUS": ["steps", "benchmark_release_runbook", "knowledge_realdata_correlation_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_realdata_correlation_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_realdata_correlation_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_realdata_correlation_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_realdata_correlation_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_matrix_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_domain_matrix_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_matrix_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_domain_matrix_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_matrix_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_matrix_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_matrix_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_matrix_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_matrix_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_matrix_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_action_plan_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_action_plan_actions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_total_action_count"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_gate_open"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_recommended_first_actions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_readiness_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_release_blockers"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_resolved_release_blockers"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_new_release_blockers"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_control_plane_drift_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_STATUS": ["steps", "benchmark_release_runbook", "knowledge_source_coverage_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_source_coverage_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES": ["steps", "benchmark_release_runbook", "knowledge_source_coverage_expansion_candidates"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_source_coverage_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS": ["steps", "benchmark_release_runbook", "knowledge_source_action_plan_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_source_action_plan_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS": ["steps", "benchmark_release_runbook", "knowledge_source_action_plan_recommended_first_actions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS": ["steps", "benchmark_release_runbook", "knowledge_source_action_plan_source_group_action_counts"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_source_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "knowledge_source_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_source_group_regressions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_source_group_improvements"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_source_drift_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_STATUS": ["steps", "benchmark_release_runbook", "knowledge_outcome_correlation_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS": ["steps", "benchmark_release_runbook", "knowledge_outcome_correlation_focus_areas"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_outcome_correlation_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES": ["steps", "benchmark_release_runbook", "knowledge_outcome_correlation_domain_statuses"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_outcome_correlation_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_resolved_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_new_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_outcome_drift_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_INDEX_STATUS": ["steps", "benchmark_release_runbook", "competitive_surpass_index_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_PRIMARY_GAPS": ["steps", "benchmark_release_runbook", "competitive_surpass_primary_gaps"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "competitive_surpass_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_STATUS": ["steps", "benchmark_release_runbook", "competitive_surpass_trend_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_SUMMARY": ["steps", "benchmark_release_runbook", "competitive_surpass_trend_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "competitive_surpass_trend_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS": ["steps", "benchmark_release_runbook", "competitive_surpass_action_plan_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT": ["steps", "benchmark_release_runbook", "competitive_surpass_action_plan_total_action_count"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS": ["steps", "benchmark_release_runbook", "competitive_surpass_action_plan_priority_pillars"], "STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "competitive_surpass_action_plan_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_ENGINEERING_STATUS": ["steps", "benchmark_release_runbook", "engineering_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_STATUS": ["steps", "benchmark_release_runbook", "realdata_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_SCORECARD_STATUS": ["steps", "benchmark_release_runbook", "realdata_scorecard_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_SCORECARD_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "realdata_scorecard_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "realdata_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_runbook", "operator_adoption_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "operator_adoption_knowledge_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "operator_adoption_knowledge_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_runbook", "scorecard_operator_adoption_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_MODE": ["steps", "benchmark_release_runbook", "scorecard_operator_adoption_mode"], "STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "scorecard_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "scorecard_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_STATUS": ["steps", "benchmark_release_runbook", "operational_operator_adoption_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "operational_operator_adoption_knowledge_outcome_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_release_runbook", "operational_operator_adoption_knowledge_outcome_drift_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_release_runbook", "operator_adoption_release_surface_alignment_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_release_runbook", "operator_adoption_release_surface_alignment_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_release_runbook", "operator_adoption_release_surface_alignment_mismatches"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_gate_open"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_releasable_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_blocked_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_blocking_reasons"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_release_gate_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_STATUS": ["steps", "benchmark_release_runbook", "knowledge_reference_inventory_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_SUMMARY": ["steps", "benchmark_release_runbook", "knowledge_reference_inventory_summary"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS": ["steps", "benchmark_release_runbook", "knowledge_reference_inventory_priority_domains"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS": ["steps", "benchmark_release_runbook", "knowledge_reference_inventory_total_reference_items"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_reference_inventory_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_OUTPUT_MD": ["steps", "benchmark_release_runbook", "output_md"], "STEP_BENCHMARK_OPERATOR_ADOPTION_ENABLED": ["steps", "benchmark_operator_adoption", "enabled"], "STEP_BENCHMARK_OPERATOR_ADOPTION_ADOPTION_READINESS": ["steps", "benchmark_operator_adoption", "adoption_readiness"], "STEP_BENCHMARK_OPERATOR_ADOPTION_OPERATOR_MODE": ["steps", "benchmark_operator_adoption", "operator_mode"], "STEP_BENCHMARK_OPERATOR_ADOPTION_NEXT_ACTION": ["steps", "benchmark_operator_adoption", "next_action"], "STEP_BENCHMARK_OPERATOR_ADOPTION_AUTOMATION_READY": ["steps", "benchmark_operator_adoption", "automation_ready"], "STEP_BENCHMARK_OPERATOR_ADOPTION_FREEZE_READY": ["steps", "benchmark_operator_adoption", "freeze_ready"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_STATUS": ["steps", "benchmark_operator_adoption", "release_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RUNBOOK_STATUS": ["steps", "benchmark_operator_adoption", "runbook_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_REVIEW_QUEUE_STATUS": ["steps", "benchmark_operator_adoption", "review_queue_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_FEEDBACK_STATUS": ["steps", "benchmark_operator_adoption", "feedback_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS": ["steps", "benchmark_operator_adoption", "knowledge_drift_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY": ["steps", "benchmark_operator_adoption", "knowledge_drift_summary"], "STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS": ["steps", "benchmark_operator_adoption", "knowledge_outcome_drift_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY": ["steps", "benchmark_operator_adoption", "knowledge_outcome_drift_summary"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS": ["steps", "benchmark_operator_adoption", "release_surface_alignment_status"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY": ["steps", "benchmark_operator_adoption", "release_surface_alignment_summary"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES": ["steps", "benchmark_operator_adoption", "release_surface_alignment_mismatches"], "STEP_BENCHMARK_OPERATOR_ADOPTION_BLOCKING_SIGNALS": ["steps", "benchmark_operator_adoption", "blocking_signals"], "STEP_BENCHMARK_OPERATOR_ADOPTION_RECOMMENDED_ACTIONS": ["steps", "benchmark_operator_adoption", "recommended_actions"], "STEP_BENCHMARK_OPERATOR_ADOPTION_OUTPUT_MD": ["steps", "benchmark_operator_adoption", "output_md"], "STEP_ASSISTANT_EVIDENCE_REPORT_ENABLED": ["steps", "assistant_evidence_report", "enabled"], "STEP_ASSISTANT_EVIDENCE_REPORT_INPUT_PATH": ["steps", "assistant_evidence_report", "input_path"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOTAL_RECORDS": ["steps", "assistant_evidence_report", "total_records"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOTAL_EVIDENCE_ITEMS": ["steps", "assistant_evidence_report", "total_evidence_items"], "STEP_ASSISTANT_EVIDENCE_REPORT_AVERAGE_EVIDENCE_COUNT": ["steps", "assistant_evidence_report", "average_evidence_count"], "STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_EVIDENCE_PCT": ["steps", "assistant_evidence_report", "records_with_evidence_pct"], "STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_DECISION_PATH_PCT": ["steps", "assistant_evidence_report", "records_with_decision_path_pct"], "STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_ANY_SOURCE_SIGNAL_PCT": ["steps", "assistant_evidence_report", "records_with_any_source_signal_pct"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOP_RECORD_KINDS": ["steps", "assistant_evidence_report", "top_record_kinds"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOP_EVIDENCE_TYPES": ["steps", "assistant_evidence_report", "top_evidence_types"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOP_STRUCTURED_SOURCES": ["steps", "assistant_evidence_report", "top_structured_sources"], "STEP_ASSISTANT_EVIDENCE_REPORT_TOP_MISSING_FIELDS": ["steps", "assistant_evidence_report", "top_missing_fields"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLED": ["steps", "active_learning_review_queue_report", "enabled"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_INPUT_PATH": ["steps", "active_learning_review_queue_report", "input_path"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOTAL": ["steps", "active_learning_review_queue_report", "total"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OPERATIONAL_STATUS": ["steps", "active_learning_review_queue_report", "operational_status"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_CRITICAL_COUNT": ["steps", "active_learning_review_queue_report", "critical_count"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_HIGH_COUNT": ["steps", "active_learning_review_queue_report", "high_count"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_AUTOMATION_READY_COUNT": ["steps", "active_learning_review_queue_report", "automation_ready_count"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_CRITICAL_RATIO": ["steps", "active_learning_review_queue_report", "critical_ratio"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_HIGH_RATIO": ["steps", "active_learning_review_queue_report", "high_ratio"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_AUTOMATION_READY_RATIO": ["steps", "active_learning_review_queue_report", "automation_ready_ratio"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_SAMPLE_TYPES": ["steps", "active_learning_review_queue_report", "top_sample_types"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_FEEDBACK_PRIORITIES": ["steps", "active_learning_review_queue_report", "top_feedback_priorities"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_DECISION_SOURCES": ["steps", "active_learning_review_queue_report", "top_decision_sources"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_REVIEW_REASONS": ["steps", "active_learning_review_queue_report", "top_review_reasons"], "STEP_OCR_REVIEW_PACK_ENABLED": ["steps", "ocr_review_pack", "enabled"], "STEP_OCR_REVIEW_PACK_INPUT_PATH": ["steps", "ocr_review_pack", "input_path"], "STEP_OCR_REVIEW_PACK_EXPORTED_RECORDS": ["steps", "ocr_review_pack", "exported_records"], "STEP_OCR_REVIEW_PACK_REVIEW_CANDIDATE_COUNT": ["steps", "ocr_review_pack", "review_candidate_count"], "STEP_OCR_REVIEW_PACK_AUTOMATION_READY_COUNT": ["steps", "ocr_review_pack", "automation_ready_count"], "STEP_OCR_REVIEW_PACK_AVERAGE_READINESS_SCORE": ["steps", "ocr_review_pack", "average_readiness_score"], "STEP_OCR_REVIEW_PACK_AVERAGE_COVERAGE_RATIO": ["steps", "ocr_review_pack", "average_coverage_ratio"], "STEP_OCR_REVIEW_PACK_REVIEW_PRIORITY_COUNTS": ["steps", "ocr_review_pack", "review_priority_counts"], "STEP_OCR_REVIEW_PACK_PRIMARY_GAP_COUNTS": ["steps", "ocr_review_pack", "primary_gap_counts"], "STEP_OCR_REVIEW_PACK_TOP_REVIEW_REASONS": ["steps", "ocr_review_pack", "top_review_reasons"], "STEP_OCR_REVIEW_PACK_TOP_RECOMMENDED_ACTIONS": ["steps", "ocr_review_pack", "top_recommended_actions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_drift_status"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_drift_domain_regressions"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_drift_domain_improvements"], "STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_artifact_bundle", "knowledge_domain_capability_drift_recommendations"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_drift_status"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_drift_domain_regressions"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_drift_domain_improvements"], "STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_companion_summary", "knowledge_domain_capability_drift_recommendations"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_drift_status"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_decision", "knowledge_domain_capability_drift_recommendations"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_drift_status"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_drift_domain_regressions"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_drift_domain_improvements"], "STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS": ["steps", "benchmark_release_runbook", "knowledge_domain_capability_drift_recommendations"]};
  for (const [envName, spec] of Object.entries(mapping)) {
    if (env[envName] !== undefined && env[envName] !== null && env[envName] !== "") {
      continue;
    }
    let value = "";
    if (spec[0] === "inputs") {
      value = inputs?.[spec[1]] ?? "";
    } else if (spec[0] === "steps") {
      value = steps?.[spec[1]]?.outputs?.[spec[2]] ?? "";
    }
    env[envName] = String(value ?? "");
  }
}

async function commentEvaluationReportPR({ github, context, core, process }) {
  loadContextEnv(process);
  try {
    const fs = require('fs');
    const combined = parseFloat(envStr("STEP_EVALUATION_COMBINED_SCORE", "") || '0');
    const vision = parseFloat(envStr("STEP_EVALUATION_VISION_SCORE", "") || '0');
    const ocr = parseFloat(envStr("STEP_EVALUATION_OCR_SCORE", "") || '0');

    const minCombined = parseFloat(envStr("WF_INPUT_MIN_COMBINED", "0.8"));
    const minVision = parseFloat(envStr("WF_INPUT_MIN_VISION", "0.65"));
    const minOcr = parseFloat(envStr("WF_INPUT_MIN_OCR", "0.9"));

    const combinedStatus = combined >= minCombined ? '✅ Pass' : '❌ Fail';
    const visionStatus = vision >= minVision ? '✅ Pass' : '❌ Fail';
    const ocrStatus = ocr >= minOcr ? '✅ Pass' : '❌ Fail';

    const overallStatus = (combined >= minCombined && vision >= minVision && ocr >= minOcr)
      ? '✅ **All checks passed!**'
      : '⚠️ **Some checks failed - review required**';

    const hasAnomalies = envStr("STEP_INSIGHTS_HAS_ANOMALIES", "") === 'true';
    const securityStatus = envStr("STEP_SECURITY_SECURITY_STATUS", "");
    const reviewPackEnabled = envStr("STEP_GRAPH2D_REVIEW_PACK_ENABLED", "") === 'true';
    const reviewGateEnabled = envStr("STEP_GRAPH2D_REVIEW_GATE_ENABLED", "") === 'true';
    const trainSweepEnabled = envStr("STEP_GRAPH2D_TRAIN_SWEEP_ENABLED", "") === 'true';
    const reviewInputCsv = envStr("STEP_GRAPH2D_REVIEW_PACK_INPUT_CSV", "");
    const reviewInputSource = envStr("STEP_GRAPH2D_REVIEW_PACK_INPUT_SOURCE", "");
    const reviewCandidates = envStr("STEP_GRAPH2D_REVIEW_PACK_CANDIDATE_ROWS", "");
    const reviewRejected = envStr("STEP_GRAPH2D_REVIEW_PACK_HYBRID_REJECTED_COUNT", "");
    const reviewConflicts = envStr("STEP_GRAPH2D_REVIEW_PACK_CONFLICT_COUNT", "");
    const reviewTopReasons = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_REVIEW_REASONS", "");
    const reviewTopPriorities = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_REVIEW_PRIORITIES", "");
    const reviewTopConfidenceBands = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_CONFIDENCE_BANDS", "");
    const reviewTopSources = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_PRIMARY_SOURCES", "");
    const reviewTopShadowSources = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_SHADOW_SOURCES", "");
    const reviewTopKnowledgeCategories = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_KNOWLEDGE_CHECK_CATEGORIES", "");
    const reviewTopStandardTypes = envStr("STEP_GRAPH2D_REVIEW_PACK_TOP_STANDARD_CANDIDATE_TYPES", "");
    const reviewExampleExplanations = envStr("STEP_GRAPH2D_REVIEW_PACK_SAMPLE_EXPLANATIONS", "");
    const reviewGateStatusRaw = envStr("STEP_GRAPH2D_REVIEW_GATE_STATUS", "");
    const reviewGateExitCode = envStr("STEP_GRAPH2D_REVIEW_GATE_EXIT_CODE", "");
    const reviewGateHeadline = envStr("STEP_GRAPH2D_REVIEW_GATE_HEADLINE", "");
    const reviewGateStrictMode = envStr("STEP_GRAPH2D_REVIEW_GATE_STRICT_STRICT_MODE", "");
    const reviewGateStrictShouldFail = envStr("STEP_GRAPH2D_REVIEW_GATE_STRICT_SHOULD_FAIL", "");
    const reviewGateStrictReason = envStr("STEP_GRAPH2D_REVIEW_GATE_STRICT_REASON", "");
    const sweepTotalRuns = envStr("STEP_GRAPH2D_TRAIN_SWEEP_TOTAL_RUNS", "");
    const sweepFailedRuns = envStr("STEP_GRAPH2D_TRAIN_SWEEP_FAILED_RUNS", "");
    const sweepBestRecipe = envStr("STEP_GRAPH2D_TRAIN_SWEEP_BEST_RECIPE", "");
    const sweepBestSeed = envStr("STEP_GRAPH2D_TRAIN_SWEEP_BEST_SEED", "");
    const sweepRecommendedEnv = envStr("STEP_GRAPH2D_TRAIN_SWEEP_RECOMMENDED_ENV_FILE", "");
    const sweepBestRunScript = envStr("STEP_GRAPH2D_TRAIN_SWEEP_BEST_RUN_SCRIPT", "");
    const benchmarkScorecardEnabled = envStr("STEP_BENCHMARK_SCORECARD_ENABLED", "") === 'true';
    const benchmarkOverallStatus = envStr("STEP_BENCHMARK_SCORECARD_OVERALL_STATUS", "");
    const benchmarkHybridStatus = envStr("STEP_BENCHMARK_SCORECARD_HYBRID_STATUS", "");
    const benchmarkGraph2dStatus = envStr("STEP_BENCHMARK_SCORECARD_GRAPH2D_STATUS", "");
    const benchmarkHistoryStatus = envStr("STEP_BENCHMARK_SCORECARD_HISTORY_STATUS", "");
    const benchmarkBrepStatus = envStr("STEP_BENCHMARK_SCORECARD_BREP_STATUS", "");
    const benchmarkGovernanceStatus = envStr("STEP_BENCHMARK_SCORECARD_GOVERNANCE_STATUS", "");
    const benchmarkAssistantStatus = envStr("STEP_BENCHMARK_SCORECARD_ASSISTANT_STATUS", "");
    const benchmarkReviewQueueStatus = envStr("STEP_BENCHMARK_SCORECARD_REVIEW_QUEUE_STATUS", "");
    const benchmarkFeedbackFlywheelStatus = envStr("STEP_BENCHMARK_SCORECARD_FEEDBACK_FLYWHEEL_STATUS", "");
    const benchmarkOcrStatus = envStr("STEP_BENCHMARK_SCORECARD_OCR_STATUS", "");
    const benchmarkQdrantStatus = envStr("STEP_BENCHMARK_SCORECARD_QDRANT_STATUS", "");
    const benchmarkKnowledgeStatus = envStr("STEP_BENCHMARK_SCORECARD_KNOWLEDGE_STATUS", "");
    const benchmarkKnowledgeReferenceItems = envStr("STEP_BENCHMARK_SCORECARD_KNOWLEDGE_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkKnowledgeFocusAreasScorecard = envStr("STEP_BENCHMARK_SCORECARD_KNOWLEDGE_FOCUS_AREAS", "");
    const benchmarkEngineeringStatus = envStr("STEP_BENCHMARK_SCORECARD_ENGINEERING_STATUS", "");
    const benchmarkEngineeringCoverageRatio = envStr("STEP_BENCHMARK_SCORECARD_ENGINEERING_COVERAGE_RATIO", "");
    const benchmarkEngineeringTopStandardTypes = envStr("STEP_BENCHMARK_SCORECARD_ENGINEERING_TOP_STANDARD_TYPES", "");
    const benchmarkScorecardOperatorAdoptionStatus = envStr("STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkScorecardOperatorAdoptionMode = envStr("STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_MODE", "");
    const benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkRecommendations = envStr("STEP_BENCHMARK_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkEngineeringEnabled = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_ENABLED", "") === 'true';
    const benchmarkEngineeringArtifactStatus = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_STATUS", "");
    const benchmarkEngineeringArtifactCoverageRatio = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_COVERAGE_RATIO", "");
    const benchmarkEngineeringRowsWithViolations = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_ROWS_WITH_VIOLATIONS", "");
    const benchmarkEngineeringRowsWithStandards = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_ROWS_WITH_STANDARDS_CANDIDATES", "");
    const benchmarkEngineeringOcrStandards = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_OCR_STANDARD_SIGNAL_COUNT", "");
    const benchmarkEngineeringRecommendations = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_RECOMMENDATIONS", "");
    const benchmarkEngineeringArtifact = envStr("STEP_BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_MD", "");
    const benchmarkRealdataEnabled = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_ENABLED", "") === 'true';
    const benchmarkRealdataStatus = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_STATUS", "");
    const benchmarkRealdataReadyComponents = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_READY_COMPONENT_COUNT", "");
    const benchmarkRealdataPartialComponents = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_PARTIAL_COMPONENT_COUNT", "");
    const benchmarkRealdataEnvironmentBlocked = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_ENVIRONMENT_BLOCKED_COUNT", "");
    const benchmarkRealdataAvailableComponents = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_AVAILABLE_COMPONENT_COUNT", "");
    const benchmarkRealdataHybridStatus = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_HYBRID_DXF_STATUS", "");
    const benchmarkRealdataHistoryStatus = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_HISTORY_H5_STATUS", "");
    const benchmarkRealdataStepSmokeStatus = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_STEP_SMOKE_STATUS", "");
    const benchmarkRealdataStepDirStatus = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_STEP_DIR_STATUS", "");
    const benchmarkRealdataRecommendations = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_RECOMMENDATIONS", "");
    const benchmarkRealdataArtifact = envStr("STEP_BENCHMARK_REALDATA_SIGNALS_OUTPUT_MD", "");
    const benchmarkRealdataScorecardEnabled = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_ENABLED", "") === 'true';
    const benchmarkRealdataScorecardStatus = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_STATUS", "");
    const benchmarkRealdataScorecardReadyComponents = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_READY_COMPONENT_COUNT", "");
    const benchmarkRealdataScorecardPartialComponents = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_PARTIAL_COMPONENT_COUNT", "");
    const benchmarkRealdataScorecardEnvironmentBlocked = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_ENVIRONMENT_BLOCKED_COUNT", "");
    const benchmarkRealdataScorecardAvailableComponents = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_AVAILABLE_COMPONENT_COUNT", "");
    const benchmarkRealdataScorecardBestSurface = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_BEST_SURFACE", "");
    const benchmarkRealdataScorecardHybridStatus = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_HYBRID_DXF_STATUS", "");
    const benchmarkRealdataScorecardHistoryStatus = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_HISTORY_H5_STATUS", "");
    const benchmarkRealdataScorecardStepSmokeStatus = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_STEP_SMOKE_STATUS", "");
    const benchmarkRealdataScorecardStepDirStatus = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_STEP_DIR_STATUS", "");
    const benchmarkRealdataScorecardRecommendations = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkRealdataScorecardArtifact = envStr("STEP_BENCHMARK_REALDATA_SCORECARD_OUTPUT_MD", "");
    const benchmarkCompetitiveSurpassEnabled = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_ENABLED", "") === 'true';
    const benchmarkCompetitiveSurpassStatus = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_STATUS", "");
    const benchmarkCompetitiveSurpassScore = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_SCORE", "");
    const benchmarkCompetitiveSurpassReadyPillars = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_READY_PILLARS", "");
    const benchmarkCompetitiveSurpassPartialPillars = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_PARTIAL_PILLARS", "");
    const benchmarkCompetitiveSurpassBlockedPillars = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_BLOCKED_PILLARS", "");
    const benchmarkCompetitiveSurpassPrimaryGaps = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_PRIMARY_GAPS", "");
    const benchmarkCompetitiveSurpassRecommendations = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_RECOMMENDATIONS", "");
    const benchmarkCompetitiveSurpassArtifact = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_MD", "");
    const benchmarkCompetitiveSurpassTrendEnabled = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_ENABLED", "") === 'true';
    const benchmarkCompetitiveSurpassTrendStatus = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_STATUS", "");
    const benchmarkCompetitiveSurpassTrendScoreDelta = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_SCORE_DELTA", "");
    const benchmarkCompetitiveSurpassTrendPillarImprovements = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_PILLAR_IMPROVEMENTS", "");
    const benchmarkCompetitiveSurpassTrendPillarRegressions = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_PILLAR_REGRESSIONS", "");
    const benchmarkCompetitiveSurpassTrendResolvedPrimaryGaps = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_RESOLVED_PRIMARY_GAPS", "");
    const benchmarkCompetitiveSurpassTrendNewPrimaryGaps = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_NEW_PRIMARY_GAPS", "");
    const benchmarkCompetitiveSurpassTrendRecommendations = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS", "");
    const benchmarkCompetitiveSurpassTrendArtifact = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_OUTPUT_MD", "");
    const benchmarkCompetitiveSurpassActionPlanEnabled = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_ENABLED", "") === 'true';
    const benchmarkCompetitiveSurpassActionPlanStatus = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS", "");
    const benchmarkCompetitiveSurpassActionPlanTotalActionCount = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkCompetitiveSurpassActionPlanHighPriorityActionCount = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkCompetitiveSurpassActionPlanMediumPriorityActionCount = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT", "");
    const benchmarkCompetitiveSurpassActionPlanPriorityPillars = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS", "");
    const benchmarkCompetitiveSurpassActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkCompetitiveSurpassActionPlanRecommendations = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkCompetitiveSurpassActionPlanArtifact = envStr("STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_OUTPUT_MD", "");
    const benchmarkKnowledgeEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_ENABLED", "") === 'true';
    const benchmarkKnowledgeArtifactStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_STATUS", "");
    const benchmarkKnowledgeTotalReferenceItems = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkKnowledgeReadyComponents = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_READY_COMPONENT_COUNT", "");
    const benchmarkKnowledgePartialComponents = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_PARTIAL_COMPONENT_COUNT", "");
    const benchmarkKnowledgeMissingComponents = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_MISSING_COMPONENT_COUNT", "");
    const benchmarkKnowledgeFocusAreaCount = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_FOCUS_AREA_COUNT", "");
    const benchmarkKnowledgeFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_FOCUS_AREAS", "");
    const benchmarkKnowledgeDomainCount = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_DOMAIN_COUNT", "");
    const benchmarkKnowledgePriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_DOMAIN_FOCUS_AREAS", "");
    const benchmarkKnowledgeRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_RECOMMENDATIONS", "");
    const benchmarkKnowledgeArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_MD", "");
    const benchmarkKnowledgeDriftEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_ENABLED", "") === 'true';
    const benchmarkKnowledgeDriftStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkKnowledgeDriftCurrentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_CURRENT_STATUS", "");
    const benchmarkKnowledgeDriftPreviousStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_PREVIOUS_STATUS", "");
    const benchmarkKnowledgeDriftReferenceItemDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_REFERENCE_ITEM_DELTA", "");
    const benchmarkKnowledgeDriftRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_REGRESSIONS", "");
    const benchmarkKnowledgeDriftImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_IMPROVEMENTS", "");
    const benchmarkKnowledgeDriftResolvedFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_RESOLVED_FOCUS_AREAS", "");
    const benchmarkKnowledgeDriftNewFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_NEW_FOCUS_AREAS", "");
    const benchmarkKnowledgeDriftDomainRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkKnowledgeDriftDomainImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkKnowledgeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDriftRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDriftArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_MD", "");
    const benchmarkKnowledgeApplicationEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_ENABLED", "") === 'true';
    const benchmarkKnowledgeApplicationStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_STATUS", "");
    const benchmarkKnowledgeApplicationReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeApplicationPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeApplicationMissingDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_MISSING_DOMAIN_COUNT", "");
    const benchmarkKnowledgeApplicationTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeApplicationFocusAreaCount = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_FOCUS_AREA_COUNT", "");
    const benchmarkKnowledgeApplicationFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_FOCUS_AREAS", "");
    const benchmarkKnowledgeApplicationPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeApplicationDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeApplicationRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_RECOMMENDATIONS", "");
    const benchmarkKnowledgeApplicationArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_MD", "");
    const benchmarkKnowledgeRealdataCorrelationEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_ENABLED", "") === 'true';
    const benchmarkKnowledgeRealdataCorrelationStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_STATUS", "");
    const benchmarkKnowledgeRealdataCorrelationReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeRealdataCorrelationPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeRealdataCorrelationBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeRealdataCorrelationTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeRealdataCorrelationFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS", "");
    const benchmarkKnowledgeRealdataCorrelationPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeRealdataCorrelationDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeRealdataCorrelationRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkKnowledgeRealdataCorrelationArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainMatrixEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainMatrixStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_STATUS", "");
    const benchmarkKnowledgeDomainMatrixReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainMatrixPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainMatrixBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainMatrixTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainMatrixFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS", "");
    const benchmarkKnowledgeDomainMatrixPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainMatrixDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeDomainMatrixRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainMatrixArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainCapabilityMatrixEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainCapabilityMatrixStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainCapabilityMatrixPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainCapabilityMatrixBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainCapabilityMatrixTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainCapabilityMatrixFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixProviderGapDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PROVIDER_GAP_DOMAINS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixSurfaceGapDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_SURFACE_GAP_DOMAINS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeDomainCapabilityMatrixRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainCapabilityMatrixArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainApiSurfaceMatrixStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_STATUS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixTotalApiRoutes = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_TOTAL_API_ROUTE_COUNT", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_FOCUS_AREAS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixPublicApiGapDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_PUBLIC_API_GAP_DOMAINS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixReferenceGapDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_REFERENCE_GAP_DOMAINS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainApiSurfaceMatrixArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainCapabilityDriftEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainCapabilityDriftStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS", "");
    const benchmarkKnowledgeDomainCapabilityDriftCurrentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_CURRENT_STATUS", "");
    const benchmarkKnowledgeDomainCapabilityDriftPreviousStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_PREVIOUS_STATUS", "");
    const benchmarkKnowledgeDomainCapabilityDriftProviderGapDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_PROVIDER_GAP_DELTA", "");
    const benchmarkKnowledgeDomainCapabilityDriftSurfaceGapDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_SURFACE_GAP_DELTA", "");
    const benchmarkKnowledgeDomainCapabilityDriftDomainRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkKnowledgeDomainCapabilityDriftDomainImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkKnowledgeDomainCapabilityDriftRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainCapabilityDriftArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainActionPlanEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainActionPlanStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS", "");
    const benchmarkKnowledgeDomainActionPlanReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanTotalActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanHighPriorityActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanMediumPriorityActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainActionPlanPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkKnowledgeDomainActionPlanDomainActionCounts = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_DOMAIN_ACTION_COUNTS", "");
    const benchmarkKnowledgeDomainActionPlanRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainActionPlanArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainSurfaceActionPlanStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_STATUS", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanTotalSubcapabilityCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_TOTAL_SUBCAPABILITY_COUNT", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanTotalActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanHighPriorityActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanMediumPriorityActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanDomainActionCounts = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_DOMAIN_ACTION_COUNTS", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainSurfaceActionPlanArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanTotalActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanHighPriorityActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanMediumPriorityActionCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanGateOpen = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainControlPlaneEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainControlPlaneStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS", "");
    const benchmarkKnowledgeDomainControlPlaneReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainControlPlanePartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneMissingDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_MISSING_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneTotalActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_TOTAL_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneHighPriorityActions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeDomainControlPlaneReleaseBlockers = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS", "");
    const benchmarkKnowledgeDomainControlPlanePriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainControlPlaneFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_FOCUS_AREAS", "");
    const benchmarkKnowledgeDomainControlPlaneRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainControlPlaneArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainControlPlaneDriftEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainControlPlaneDriftStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftCurrentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_CURRENT_STATUS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftPreviousStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_PREVIOUS_STATUS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftReadyDomainDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_READY_DOMAIN_DELTA", "");
    const benchmarkKnowledgeDomainControlPlaneDriftBlockedDomainDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_BLOCKED_DOMAIN_DELTA", "");
    const benchmarkKnowledgeDomainControlPlaneDriftTotalActionDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_TOTAL_ACTION_DELTA", "");
    const benchmarkKnowledgeDomainControlPlaneDriftHighPriorityActionDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_HIGH_PRIORITY_ACTION_DELTA", "");
    const benchmarkKnowledgeDomainControlPlaneDriftDomainRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftDomainImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftNewReleaseBlockers = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainControlPlaneDriftArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainReleaseGateEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainReleaseGateStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS", "");
    const benchmarkKnowledgeDomainReleaseGateGateOpen = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN", "");
    const benchmarkKnowledgeDomainReleaseGateReleasableDomainCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainReleaseGateBlockedDomainCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainReleaseGatePartialDomainCount = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeDomainReleaseGateBlockingReasons = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS", "");
    const benchmarkKnowledgeDomainReleaseGateWarningReasons = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_WARNING_REASONS", "");
    const benchmarkKnowledgeDomainReleaseGateReleasableDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS", "");
    const benchmarkKnowledgeDomainReleaseGateBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS", "");
    const benchmarkKnowledgeDomainReleaseGatePriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeDomainReleaseGateRecommendedFirstAction = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDED_FIRST_ACTION", "");
    const benchmarkKnowledgeDomainReleaseGateRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS", "");
    const benchmarkKnowledgeDomainReleaseGateArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_OUTPUT_MD", "");
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_ENABLED", "") === 'true';
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_OUTPUT_MD", "");
    const benchmarkKnowledgeReferenceInventoryEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_ENABLED", "") === 'true';
    const benchmarkKnowledgeReferenceInventoryStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_STATUS", "");
    const benchmarkKnowledgeReferenceInventoryReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryTotalReferenceItems = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkKnowledgeReferenceInventoryTotalTables = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_TABLE_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryPopulatedTables = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_POPULATED_TABLE_COUNT", "");
    const benchmarkKnowledgeReferenceInventoryPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeReferenceInventoryFocusTables = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_FOCUS_TABLES", "");
    const benchmarkKnowledgeReferenceInventoryRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS", "");
    const benchmarkKnowledgeReferenceInventoryArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_MD", "");
    const benchmarkKnowledgeSourceCoverageEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_ENABLED", "") === 'true';
    const benchmarkKnowledgeSourceCoverageStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_STATUS", "");
    const benchmarkKnowledgeSourceCoverageReadySourceGroups = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_READY_SOURCE_GROUP_COUNT", "");
    const benchmarkKnowledgeSourceCoveragePartialSourceGroups = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_PARTIAL_SOURCE_GROUP_COUNT", "");
    const benchmarkKnowledgeSourceCoverageMissingSourceGroups = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_MISSING_SOURCE_GROUP_COUNT", "");
    const benchmarkKnowledgeSourceCoverageTotalSourceGroups = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_GROUP_COUNT", "");
    const benchmarkKnowledgeSourceCoverageTotalSourceTables = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_TABLE_COUNT", "");
    const benchmarkKnowledgeSourceCoverageTotalSourceItems = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_SOURCE_ITEM_COUNT", "");
    const benchmarkKnowledgeSourceCoverageTotalReferenceStandards = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_TOTAL_REFERENCE_STANDARD_COUNT", "");
    const benchmarkKnowledgeSourceCoverageReadyExpansionCandidates = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_READY_EXPANSION_CANDIDATE_COUNT", "");
    const benchmarkKnowledgeSourceCoverageFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_FOCUS_AREAS", "");
    const benchmarkKnowledgeSourceCoveragePriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeSourceCoverageDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeSourceCoverageExpansionCandidates = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES", "");
    const benchmarkKnowledgeSourceCoverageRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS", "");
    const benchmarkKnowledgeSourceCoverageArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_OUTPUT_MD", "");
    const benchmarkKnowledgeSourceActionPlanEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_ENABLED", "") === 'true';
    const benchmarkKnowledgeSourceActionPlanStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS", "");
    const benchmarkKnowledgeSourceActionPlanTotalActions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkKnowledgeSourceActionPlanHighPriorityActions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_HIGH_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeSourceActionPlanMediumPriorityActions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_MEDIUM_PRIORITY_ACTION_COUNT", "");
    const benchmarkKnowledgeSourceActionPlanExpansionActions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_EXPANSION_ACTION_COUNT", "");
    const benchmarkKnowledgeSourceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeSourceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkKnowledgeSourceActionPlanSourceGroupActionCounts = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS", "");
    const benchmarkKnowledgeSourceActionPlanRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkKnowledgeSourceActionPlanArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_MD", "");
    const benchmarkKnowledgeSourceDriftEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_ENABLED", "") === 'true';
    const benchmarkKnowledgeSourceDriftStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_STATUS", "");
    const benchmarkKnowledgeSourceDriftCurrentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_CURRENT_STATUS", "");
    const benchmarkKnowledgeSourceDriftPreviousStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_PREVIOUS_STATUS", "");
    const benchmarkKnowledgeSourceDriftReadySourceGroupDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_READY_SOURCE_GROUP_DELTA", "");
    const benchmarkKnowledgeSourceDriftMissingSourceGroupDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_MISSING_SOURCE_GROUP_DELTA", "");
    const benchmarkKnowledgeSourceDriftRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_REGRESSIONS", "");
    const benchmarkKnowledgeSourceDriftImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_IMPROVEMENTS", "");
    const benchmarkKnowledgeSourceDriftSourceGroupRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS", "");
    const benchmarkKnowledgeSourceDriftSourceGroupImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS", "");
    const benchmarkKnowledgeSourceDriftResolvedFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_FOCUS_AREAS", "");
    const benchmarkKnowledgeSourceDriftNewFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_NEW_FOCUS_AREAS", "");
    const benchmarkKnowledgeSourceDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeSourceDriftNewPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeSourceDriftRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkKnowledgeSourceDriftArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_OUTPUT_MD", "");
    const benchmarkKnowledgeOutcomeCorrelationEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_ENABLED", "") === 'true';
    const benchmarkKnowledgeOutcomeCorrelationStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_STATUS", "");
    const benchmarkKnowledgeOutcomeCorrelationReadyDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_READY_DOMAIN_COUNT", "");
    const benchmarkKnowledgeOutcomeCorrelationPartialDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_PARTIAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeOutcomeCorrelationBlockedDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_BLOCKED_DOMAIN_COUNT", "");
    const benchmarkKnowledgeOutcomeCorrelationTotalDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_TOTAL_DOMAIN_COUNT", "");
    const benchmarkKnowledgeOutcomeCorrelationFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS", "");
    const benchmarkKnowledgeOutcomeCorrelationPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeOutcomeCorrelationDomainStatuses = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkKnowledgeOutcomeCorrelationRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkKnowledgeOutcomeCorrelationArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_OUTPUT_MD", "");
    const benchmarkKnowledgeOutcomeDriftEnabled = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_ENABLED", "") === 'true';
    const benchmarkKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkKnowledgeOutcomeDriftCurrentStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_CURRENT_STATUS", "");
    const benchmarkKnowledgeOutcomeDriftPreviousStatus = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_PREVIOUS_STATUS", "");
    const benchmarkKnowledgeOutcomeDriftReadyDomainDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_READY_DOMAIN_DELTA", "");
    const benchmarkKnowledgeOutcomeDriftBlockedDomainDelta = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_BLOCKED_DOMAIN_DELTA", "");
    const benchmarkKnowledgeOutcomeDriftRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_REGRESSIONS", "");
    const benchmarkKnowledgeOutcomeDriftImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_IMPROVEMENTS", "");
    const benchmarkKnowledgeOutcomeDriftDomainRegressions = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkKnowledgeOutcomeDriftDomainImprovements = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkKnowledgeOutcomeDriftResolvedFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_FOCUS_AREAS", "");
    const benchmarkKnowledgeOutcomeDriftNewFocusAreas = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_NEW_FOCUS_AREAS", "");
    const benchmarkKnowledgeOutcomeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeOutcomeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkKnowledgeOutcomeDriftRecommendations = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS", "");
    const benchmarkKnowledgeOutcomeDriftArtifact = envStr("STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_OUTPUT_MD", "");
    const feedbackFlywheelBenchmarkEnabled = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_ENABLED", "") === 'true';
    const feedbackFlywheelBenchmarkStatus = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_STATUS", "");
    const feedbackFlywheelBenchmarkTotal = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_FEEDBACK_TOTAL", "");
    const feedbackFlywheelBenchmarkCorrections = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_CORRECTION_COUNT", "");
    const feedbackFlywheelBenchmarkFinetuneSamples = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_FINETUNE_SAMPLE_COUNT", "");
    const feedbackFlywheelBenchmarkMetricTriplets = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_METRIC_TRIPLET_COUNT", "");
    const feedbackFlywheelBenchmarkArtifact = envStr("STEP_FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_MD", "");
    const benchmarkOperationalSummaryEnabled = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_ENABLED", "") === 'true';
    const benchmarkOperationalSummaryOverall = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OVERALL_STATUS", "");
    const benchmarkOperationalFeedbackStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_FEEDBACK_STATUS", "");
    const benchmarkOperationalAssistantStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_ASSISTANT_STATUS", "");
    const benchmarkOperationalReviewQueueStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_REVIEW_QUEUE_STATUS", "");
    const benchmarkOperationalOcrStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OCR_STATUS", "");
    const benchmarkOperationalOperatorAdoptionStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkOperationalBlockers = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_BLOCKERS", "");
    const benchmarkOperationalRecommendations = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_RECOMMENDATIONS", "");
    const benchmarkOperationalArtifact = envStr("STEP_BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_MD", "");
    const benchmarkArtifactBundleEnabled = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_ENABLED", "") === 'true';
    const benchmarkArtifactBundleOverall = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OVERALL_STATUS", "");
    const benchmarkArtifactBundleAvailableArtifacts = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_AVAILABLE_ARTIFACT_COUNT", "");
    const benchmarkArtifactBundleFeedbackStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_STATUS", "");
    const benchmarkArtifactBundleAssistantStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_STATUS", "");
    const benchmarkArtifactBundleReviewQueueStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_STATUS", "");
    const benchmarkArtifactBundleOcrStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OCR_STATUS", "");
    const benchmarkArtifactBundleKnowledgeStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleKnowledgeDriftRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDriftChanges = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_COMPONENT_CHANGES", "");
    const benchmarkArtifactBundleKnowledgeDriftDomainRegressions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkArtifactBundleKnowledgeDriftDomainImprovements = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkArtifactBundleKnowledgeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgePriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeApplicationStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_STATUS", "");
    const benchmarkArtifactBundleKnowledgeApplicationFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeApplicationPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeApplicationDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeApplicationRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_STATUS", "");
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainMatrixStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainMatrixFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeDomainMatrixPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainMatrixDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeDomainMatrixRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainActionPlanStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainActionPlanActions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainActionPlanPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainActionPlanRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanTotalActionCount = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanGateOpen = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneReleaseBlockers = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftDomainRegressions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftDomainImprovements = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftNewReleaseBlockers = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS", "");
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_SUMMARY", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateGateOpen = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateBlockingReasons = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateReleasableDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateBlockedDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGatePriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeReferenceInventoryStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_STATUS", "");
    const benchmarkArtifactBundleKnowledgeReferenceInventoryPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeReferenceInventoryTotalReferenceItems = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkArtifactBundleKnowledgeSourceCoverageStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_STATUS", "");
    const benchmarkArtifactBundleKnowledgeSourceCoverageDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeSourceCoverageExpansionCandidates = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES", "");
    const benchmarkArtifactBundleKnowledgeSourceCoverageRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeSourceActionPlanStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS", "");
    const benchmarkArtifactBundleKnowledgeSourceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeSourceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkArtifactBundleKnowledgeSourceActionPlanSourceGroupActionCounts = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS", "");
    const benchmarkArtifactBundleKnowledgeSourceActionPlanRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftSourceGroupRegressions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftSourceGroupImprovements = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftNewPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeSourceDriftRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_STATUS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationFocusAreas = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationDomainStatuses = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftDomainRegressions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftDomainImprovements = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkArtifactBundleKnowledgeOutcomeDriftRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleEngineeringStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_STATUS", "");
    const benchmarkArtifactBundleRealdataStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_STATUS", "");
    const benchmarkArtifactBundleRealdataScorecardStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_STATUS", "");
    const benchmarkArtifactBundleRealdataScorecardRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleRealdataRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkArtifactBundleOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleScorecardOperatorAdoptionStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkArtifactBundleScorecardOperatorAdoptionMode = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_MODE", "");
    const benchmarkArtifactBundleScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkArtifactBundleScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleOperationalOperatorAdoptionStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkArtifactBundleOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkArtifactBundleOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkArtifactBundleCompetitiveSurpassStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_STATUS", "");
    const benchmarkArtifactBundleCompetitiveSurpassPrimaryGaps = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_PRIMARY_GAPS", "");
    const benchmarkArtifactBundleCompetitiveSurpassRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleCompetitiveSurpassTrendStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_STATUS", "");
    const benchmarkArtifactBundleCompetitiveSurpassTrendSummary = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_SUMMARY", "");
    const benchmarkArtifactBundleCompetitiveSurpassTrendRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS", "");
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanTotalActionCount = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanPriorityPillars = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS", "");
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleBlockers = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_BLOCKERS", "");
    const benchmarkArtifactBundleRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleArtifact = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_MD", "");
    const benchmarkCompanionSummaryEnabled = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_ENABLED", "") === 'true';
    const benchmarkCompanionSummaryOverall = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OVERALL_STATUS", "");
    const benchmarkCompanionReviewSurface = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REVIEW_SURFACE", "");
    const benchmarkCompanionPrimaryGap = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_PRIMARY_GAP", "");
    const benchmarkCompanionHybridStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_HYBRID_STATUS", "");
    const benchmarkCompanionAssistantStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_ASSISTANT_STATUS", "");
    const benchmarkCompanionReviewQueueStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REVIEW_QUEUE_STATUS", "");
    const benchmarkCompanionOcrStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OCR_STATUS", "");
    const benchmarkCompanionQdrantStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_QDRANT_STATUS", "");
    const benchmarkCompanionKnowledgeStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_STATUS", "");
    const benchmarkCompanionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkCompanionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkCompanionKnowledgeDriftRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDriftChanges = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_COMPONENT_CHANGES", "");
    const benchmarkCompanionKnowledgeDriftDomainRegressions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkCompanionKnowledgeDriftDomainImprovements = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkCompanionKnowledgeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgePriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeApplicationStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_STATUS", "");
    const benchmarkCompanionKnowledgeApplicationFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeApplicationPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeApplicationDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeApplicationRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_APPLICATION_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeRealdataCorrelationStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_STATUS", "");
    const benchmarkCompanionKnowledgeRealdataCorrelationFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeRealdataCorrelationPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeRealdataCorrelationDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeRealdataCorrelationRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainMatrixStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_STATUS", "");
    const benchmarkCompanionKnowledgeDomainMatrixFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeDomainMatrixPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainMatrixDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeDomainMatrixRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainActionPlanStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS", "");
    const benchmarkCompanionKnowledgeDomainActionPlanActions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS", "");
    const benchmarkCompanionKnowledgeDomainActionPlanPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainActionPlanRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanTotalActionCount = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanGateOpen = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneReleaseBlockers = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftDomainRegressions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftDomainImprovements = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftNewReleaseBlockers = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS", "");
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_SUMMARY", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateGateOpen = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateBlockingReasons = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateReleasableDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateBlockedDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGatePriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeDomainReleaseGateRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeReferenceInventoryStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_STATUS", "");
    const benchmarkCompanionKnowledgeReferenceInventoryPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeReferenceInventoryTotalReferenceItems = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkCompanionKnowledgeSourceCoverageStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_STATUS", "");
    const benchmarkCompanionKnowledgeSourceCoverageDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeSourceCoverageExpansionCandidates = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES", "");
    const benchmarkCompanionKnowledgeSourceCoverageRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeSourceActionPlanStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS", "");
    const benchmarkCompanionKnowledgeSourceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeSourceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkCompanionKnowledgeSourceActionPlanSourceGroupActionCounts = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS", "");
    const benchmarkCompanionKnowledgeSourceActionPlanRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeSourceDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_STATUS", "");
    const benchmarkCompanionKnowledgeSourceDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SUMMARY", "");
    const benchmarkCompanionKnowledgeSourceDriftSourceGroupRegressions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS", "");
    const benchmarkCompanionKnowledgeSourceDriftSourceGroupImprovements = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS", "");
    const benchmarkCompanionKnowledgeSourceDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeSourceDriftNewPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeSourceDriftRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeOutcomeCorrelationStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_STATUS", "");
    const benchmarkCompanionKnowledgeOutcomeCorrelationFocusAreas = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS", "");
    const benchmarkCompanionKnowledgeOutcomeCorrelationPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeOutcomeCorrelationDomainStatuses = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkCompanionKnowledgeOutcomeCorrelationRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkCompanionKnowledgeOutcomeDriftDomainRegressions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftDomainImprovements = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkCompanionKnowledgeOutcomeDriftRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS", "");
    const benchmarkCompanionEngineeringStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_ENGINEERING_STATUS", "");
    const benchmarkCompanionRealdataStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_STATUS", "");
    const benchmarkCompanionRealdataScorecardStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_SCORECARD_STATUS", "");
    const benchmarkCompanionRealdataScorecardRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkCompanionRealdataRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_REALDATA_RECOMMENDATIONS", "");
    const benchmarkCompanionOperatorAdoptionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkCompanionOperatorAdoptionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkCompanionOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkCompanionOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkCompanionScorecardOperatorAdoptionStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkCompanionScorecardOperatorAdoptionMode = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_MODE", "");
    const benchmarkCompanionScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkCompanionScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkCompanionOperationalOperatorAdoptionStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkCompanionOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkCompanionOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkCompanionCompetitiveSurpassStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_INDEX_STATUS", "");
    const benchmarkCompanionCompetitiveSurpassPrimaryGaps = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_PRIMARY_GAPS", "");
    const benchmarkCompanionCompetitiveSurpassRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_RECOMMENDATIONS", "");
    const benchmarkCompanionCompetitiveSurpassTrendStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_STATUS", "");
    const benchmarkCompanionCompetitiveSurpassTrendSummary = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_SUMMARY", "");
    const benchmarkCompanionCompetitiveSurpassTrendRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS", "");
    const benchmarkCompanionCompetitiveSurpassActionPlanStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS", "");
    const benchmarkCompanionCompetitiveSurpassActionPlanTotalActionCount = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkCompanionCompetitiveSurpassActionPlanPriorityPillars = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS", "");
    const benchmarkCompanionCompetitiveSurpassActionPlanRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkCompanionBlockers = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_BLOCKERS", "");
    const benchmarkCompanionRecommendedActions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_RECOMMENDED_ACTIONS", "");
    const benchmarkCompanionArtifact = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_OUTPUT_MD", "");
    const benchmarkReleaseDecisionEnabled = envStr("STEP_BENCHMARK_RELEASE_DECISION_ENABLED", "") === 'true';
    const benchmarkReleaseStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_RELEASE_STATUS", "");
    const benchmarkReleaseAutomationReady = envStr("STEP_BENCHMARK_RELEASE_DECISION_AUTOMATION_READY", "");
    const benchmarkReleasePrimarySignalSource = envStr("STEP_BENCHMARK_RELEASE_DECISION_PRIMARY_SIGNAL_SOURCE", "");
    const benchmarkReleaseBlockingSignals = envStr("STEP_BENCHMARK_RELEASE_DECISION_BLOCKING_SIGNALS", "");
    const benchmarkReleaseReviewSignals = envStr("STEP_BENCHMARK_RELEASE_DECISION_REVIEW_SIGNALS", "");
    const benchmarkReleaseHybridStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_HYBRID_STATUS", "");
    const benchmarkReleaseAssistantStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_ASSISTANT_STATUS", "");
    const benchmarkReleaseReviewQueueStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_REVIEW_QUEUE_STATUS", "");
    const benchmarkReleaseOcrStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OCR_STATUS", "");
    const benchmarkReleaseQdrantStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_QDRANT_STATUS", "");
    const benchmarkReleaseKnowledgeStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_STATUS", "");
    const benchmarkReleaseKnowledgeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkReleaseKnowledgeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkReleaseKnowledgeDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseKnowledgeDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseKnowledgeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgePriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeApplicationStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_STATUS", "");
    const benchmarkReleaseKnowledgeApplicationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeApplicationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeApplicationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeApplicationRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_APPLICATION_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeRealdataCorrelationStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_STATUS", "");
    const benchmarkReleaseKnowledgeRealdataCorrelationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeRealdataCorrelationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeRealdataCorrelationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeRealdataCorrelationRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainMatrixStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_STATUS", "");
    const benchmarkReleaseKnowledgeDomainMatrixFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeDomainMatrixPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainMatrixDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeDomainMatrixRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseKnowledgeDomainActionPlanActions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS", "");
    const benchmarkReleaseKnowledgeDomainActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanTotalActionCount = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanGateOpen = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftNewReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS", "");
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeSourceCoverageStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_STATUS", "");
    const benchmarkReleaseKnowledgeSourceCoverageDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeSourceCoverageExpansionCandidates = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES", "");
    const benchmarkReleaseKnowledgeSourceCoverageRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeSourceActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseKnowledgeSourceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeSourceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkReleaseKnowledgeSourceActionPlanSourceGroupActionCounts = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS", "");
    const benchmarkReleaseKnowledgeSourceActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeSourceDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_STATUS", "");
    const benchmarkReleaseKnowledgeSourceDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SUMMARY", "");
    const benchmarkReleaseKnowledgeSourceDriftSourceGroupRegressions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS", "");
    const benchmarkReleaseKnowledgeSourceDriftSourceGroupImprovements = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS", "");
    const benchmarkReleaseKnowledgeSourceDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeSourceDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeSourceDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeOutcomeCorrelationStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_STATUS", "");
    const benchmarkReleaseKnowledgeOutcomeCorrelationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS", "");
    const benchmarkReleaseKnowledgeOutcomeCorrelationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeOutcomeCorrelationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseKnowledgeOutcomeCorrelationRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseKnowledgeOutcomeDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeOutcomeDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseCompetitiveSurpassStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_INDEX_STATUS", "");
    const benchmarkReleaseCompetitiveSurpassPrimaryGaps = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_PRIMARY_GAPS", "");
    const benchmarkReleaseCompetitiveSurpassRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_RECOMMENDATIONS", "");
    const benchmarkReleaseCompetitiveSurpassTrendStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_STATUS", "");
    const benchmarkReleaseCompetitiveSurpassTrendSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_SUMMARY", "");
    const benchmarkReleaseCompetitiveSurpassTrendRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS", "");
    const benchmarkReleaseCompetitiveSurpassActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseCompetitiveSurpassActionPlanTotalActionCount = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkReleaseCompetitiveSurpassActionPlanPriorityPillars = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS", "");
    const benchmarkReleaseCompetitiveSurpassActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseEngineeringStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_ENGINEERING_STATUS", "");
    const benchmarkReleaseRealdataStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_REALDATA_STATUS", "");
    const benchmarkReleaseRealdataScorecardStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_REALDATA_SCORECARD_STATUS", "");
    const benchmarkReleaseRealdataScorecardRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_REALDATA_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkReleaseRealdataRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_REALDATA_RECOMMENDATIONS", "");
    const benchmarkReleaseOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseOperatorAdoptionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkReleaseOperatorAdoptionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseScorecardOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseScorecardOperatorAdoptionMode = envStr("STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_MODE", "");
    const benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseOperationalOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_RELEASE_DECISION_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateGateOpen = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateReleasableDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateBlockedDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainReleaseGatePriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateBlockingReasons = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS", "");
    const benchmarkReleaseKnowledgeDomainReleaseGateRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeReferenceInventoryStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_STATUS", "");
    const benchmarkReleaseKnowledgeReferenceInventorySummary = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_SUMMARY", "");
    const benchmarkReleaseKnowledgeReferenceInventoryPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS", "");
    const benchmarkReleaseKnowledgeReferenceInventoryTotalReferenceItems = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkReleaseKnowledgeReferenceInventoryRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS", "");
    const benchmarkReleaseArtifact = envStr("STEP_BENCHMARK_RELEASE_DECISION_OUTPUT_MD", "");
    const benchmarkReleaseRunbookEnabled = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_ENABLED", "") === 'true';
    const benchmarkReleaseRunbookStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_RELEASE_STATUS", "");
    const benchmarkReleaseRunbookFreezeReady = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_READY_TO_FREEZE_BASELINE", "");
    const benchmarkReleaseRunbookPrimarySignalSource = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_PRIMARY_SIGNAL_SOURCE", "");
    const benchmarkReleaseRunbookNextAction = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_NEXT_ACTION", "");
    const benchmarkReleaseRunbookMissingArtifacts = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_MISSING_ARTIFACTS", "");
    const benchmarkReleaseRunbookBlockingSignals = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_BLOCKING_SIGNALS", "");
    const benchmarkReleaseRunbookReviewSignals = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_REVIEW_SIGNALS", "");
    const benchmarkReleaseRunbookKnowledgeStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookKnowledgeDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseRunbookKnowledgeDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseRunbookKnowledgeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgePriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeApplicationStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeApplicationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeApplicationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeApplicationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeApplicationRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_APPLICATION_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REALDATA_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainMatrixStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainMatrixFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeDomainMatrixPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainMatrixDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeDomainMatrixRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_MATRIX_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainActionPlanActions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_ACTIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanTotalActionCount = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanGateOpen = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_GATE_OPEN", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_READINESS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RELEASE_BLOCKERS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RESOLVED_RELEASE_BLOCKERS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftNewReleaseBlockers = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_NEW_RELEASE_BLOCKERS", "");
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeSourceCoverageStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeSourceCoverageDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeSourceCoverageExpansionCandidates = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_EXPANSION_CANDIDATES", "");
    const benchmarkReleaseRunbookKnowledgeSourceCoverageRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_COVERAGE_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanRecommendedFirstActions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDED_FIRST_ACTIONS", "");
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanSourceGroupActionCounts = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_SOURCE_GROUP_ACTION_COUNTS", "");
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftSourceGroupRegressions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_REGRESSIONS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftSourceGroupImprovements = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_SOURCE_GROUP_IMPROVEMENTS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeSourceDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_SOURCE_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationFocusAreas = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_FOCUS_AREAS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationDomainStatuses = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_DOMAIN_STATUSES", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_CORRELATION_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftResolvedPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_RESOLVED_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftNewPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_NEW_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_OUTCOME_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_INDEX_STATUS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassPrimaryGaps = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_PRIMARY_GAPS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassTrendStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_STATUS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassTrendSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_SUMMARY", "");
    const benchmarkReleaseRunbookCompetitiveSurpassTrendRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_TREND_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_STATUS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanTotalActionCount = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_TOTAL_ACTION_COUNT", "");
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanPriorityPillars = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_PRIORITY_PILLARS", "");
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_COMPETITIVE_SURPASS_ACTION_PLAN_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookEngineeringStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_ENGINEERING_STATUS", "");
    const benchmarkReleaseRunbookRealdataStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_STATUS", "");
    const benchmarkReleaseRunbookRealdataScorecardStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_SCORECARD_STATUS", "");
    const benchmarkReleaseRunbookRealdataScorecardRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_SCORECARD_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookRealdataRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_REALDATA_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookScorecardOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseRunbookScorecardOperatorAdoptionMode = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_MODE", "");
    const benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_SCORECARD_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookOperationalOperatorAdoptionStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_STATUS", "");
    const benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATIONAL_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateGateOpen = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_GATE_OPEN", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateReleasableDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RELEASABLE_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateBlockedDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKED_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGatePriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateBlockingReasons = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_BLOCKING_REASONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_RELEASE_GATE_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeReferenceInventorySummary = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_SUMMARY", "");
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryPriorityDomains = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_PRIORITY_DOMAINS", "");
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryTotalReferenceItems = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_TOTAL_REFERENCE_ITEMS", "");
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookArtifact = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_OUTPUT_MD", "");
    const benchmarkOperatorAdoptionEnabled = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_ENABLED", "") === 'true';
    const benchmarkOperatorAdoptionReadiness = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_ADOPTION_READINESS", "");
    const benchmarkOperatorAdoptionMode = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_OPERATOR_MODE", "");
    const benchmarkOperatorAdoptionNextAction = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_NEXT_ACTION", "");
    const benchmarkOperatorAdoptionAutomationReady = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_AUTOMATION_READY", "");
    const benchmarkOperatorAdoptionFreezeReady = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_FREEZE_READY", "");
    const benchmarkOperatorAdoptionReleaseStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_STATUS", "");
    const benchmarkOperatorAdoptionRunbookStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RUNBOOK_STATUS", "");
    const benchmarkOperatorAdoptionReviewQueueStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_REVIEW_QUEUE_STATUS", "");
    const benchmarkOperatorAdoptionFeedbackStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_FEEDBACK_STATUS", "");
    const benchmarkOperatorAdoptionKnowledgeDriftStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_STATUS", "");
    const benchmarkOperatorAdoptionKnowledgeDriftSummary = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_DRIFT_SUMMARY", "");
    const benchmarkOperatorAdoptionKnowledgeOutcomeDriftStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_STATUS", "");
    const benchmarkOperatorAdoptionKnowledgeOutcomeDriftSummary = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_KNOWLEDGE_OUTCOME_DRIFT_SUMMARY", "");
    const benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_STATUS", "");
    const benchmarkOperatorAdoptionReleaseSurfaceAlignmentSummary = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_SUMMARY", "");
    const benchmarkOperatorAdoptionReleaseSurfaceAlignmentMismatches = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RELEASE_SURFACE_ALIGNMENT_MISMATCHES", "");
    const benchmarkOperatorAdoptionBlockingSignals = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_BLOCKING_SIGNALS", "");
    const benchmarkOperatorAdoptionRecommendedActions = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_RECOMMENDED_ACTIONS", "");
    const benchmarkOperatorAdoptionArtifact = envStr("STEP_BENCHMARK_OPERATOR_ADOPTION_OUTPUT_MD", "");
    const assistantEvidenceEnabled = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_ENABLED", "") === 'true';
    const assistantEvidenceInput = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_INPUT_PATH", "");
    const assistantEvidenceTotalRecords = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOTAL_RECORDS", "");
    const assistantEvidenceTotalItems = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOTAL_EVIDENCE_ITEMS", "");
    const assistantEvidenceAverageCount = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_AVERAGE_EVIDENCE_COUNT", "");
    const assistantEvidenceCoverage = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_EVIDENCE_PCT", "");
    const assistantDecisionPathCoverage = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_DECISION_PATH_PCT", "");
    const assistantSourceSignalCoverage = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_RECORDS_WITH_ANY_SOURCE_SIGNAL_PCT", "");
    const assistantTopRecordKinds = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOP_RECORD_KINDS", "");
    const assistantTopEvidenceTypes = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOP_EVIDENCE_TYPES", "");
    const assistantTopStructuredSources = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOP_STRUCTURED_SOURCES", "");
    const assistantTopMissingFields = envStr("STEP_ASSISTANT_EVIDENCE_REPORT_TOP_MISSING_FIELDS", "");
    const activeLearningReviewQueueEnabled = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_ENABLED", "") === 'true';
    const activeLearningReviewQueueInput = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_INPUT_PATH", "");
    const activeLearningReviewQueueTotal = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOTAL", "");
    const activeLearningReviewQueueStatus = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OPERATIONAL_STATUS", "");
    const activeLearningReviewQueueCriticalCount = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_CRITICAL_COUNT", "");
    const activeLearningReviewQueueHighCount = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_HIGH_COUNT", "");
    const activeLearningReviewQueueAutomationReadyCount = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_AUTOMATION_READY_COUNT", "");
    const activeLearningReviewQueueCriticalRatio = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_CRITICAL_RATIO", "");
    const activeLearningReviewQueueHighRatio = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_HIGH_RATIO", "");
    const activeLearningReviewQueueAutomationReadyRatio = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_AUTOMATION_READY_RATIO", "");
    const activeLearningReviewQueueTopSampleTypes = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_SAMPLE_TYPES", "");
    const activeLearningReviewQueueTopPriorities = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_FEEDBACK_PRIORITIES", "");
    const activeLearningReviewQueueTopDecisionSources = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_DECISION_SOURCES", "");
    const activeLearningReviewQueueTopReviewReasons = envStr("STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_TOP_REVIEW_REASONS", "");
    const ocrReviewPackEnabled = envStr("STEP_OCR_REVIEW_PACK_ENABLED", "") === 'true';
    const ocrReviewPackInput = envStr("STEP_OCR_REVIEW_PACK_INPUT_PATH", "");
    const ocrExportedRecords = envStr("STEP_OCR_REVIEW_PACK_EXPORTED_RECORDS", "");
    const ocrReviewCandidateCount = envStr("STEP_OCR_REVIEW_PACK_REVIEW_CANDIDATE_COUNT", "");
    const ocrAutomationReadyCount = envStr("STEP_OCR_REVIEW_PACK_AUTOMATION_READY_COUNT", "");
    const ocrAverageReadiness = envStr("STEP_OCR_REVIEW_PACK_AVERAGE_READINESS_SCORE", "");
    const ocrAverageCoverage = envStr("STEP_OCR_REVIEW_PACK_AVERAGE_COVERAGE_RATIO", "");
    const ocrReviewPriorities = envStr("STEP_OCR_REVIEW_PACK_REVIEW_PRIORITY_COUNTS", "");
    const ocrPrimaryGaps = envStr("STEP_OCR_REVIEW_PACK_PRIMARY_GAP_COUNTS", "");
    const ocrReviewReasons = envStr("STEP_OCR_REVIEW_PACK_TOP_REVIEW_REASONS", "");
    const ocrRecommendedActions = envStr("STEP_OCR_REVIEW_PACK_TOP_RECOMMENDED_ACTIONS", "");
    const reviewCandidateCount = parseInt(reviewCandidates || '0', 10);
    const sweepTotalRunsInt = parseInt(sweepTotalRuns || '0', 10);
    const sweepFailedRunsInt = parseInt(sweepFailedRuns || '0', 10);

    const reviewPackStatus = reviewPackEnabled
      ? `🧪 input=${reviewInputCsv} (source=${reviewInputSource || 'unknown'}), candidates=${reviewCandidates}, rejected=${reviewRejected}, conflict=${reviewConflicts}`
      : '⏭️ skipped';
    const reviewPackInsights = reviewPackEnabled
      ? `reasons=${reviewTopReasons || 'n/a'}, priorities=${reviewTopPriorities || 'n/a'}, bands=${reviewTopConfidenceBands || 'n/a'}, knowledge=${reviewTopKnowledgeCategories || 'n/a'}, standards=${reviewTopStandardTypes || 'n/a'}, sources=${reviewTopSources || 'n/a'}, shadow=${reviewTopShadowSources || 'n/a'}, examples=${reviewExampleExplanations || 'n/a'}`
      : '⏭️ skipped';
    const reviewGateStatus = reviewGateEnabled
      ? `${reviewGateStatusRaw} (exit=${reviewGateExitCode}, headline=${reviewGateHeadline || 'n/a'})`
      : '⏭️ skipped';
    const reviewGateStrictStatus = reviewGateEnabled
      ? `strict=${reviewGateStrictMode || 'false'}, should_fail=${reviewGateStrictShouldFail || 'false'}, reason=${reviewGateStrictReason || 'n/a'}`
      : '⏭️ skipped';
    const trainSweepStatus = trainSweepEnabled
      ? `runs=${sweepTotalRuns}, failed=${sweepFailedRuns}, best=${sweepBestRecipe}@${sweepBestSeed}, env=${sweepRecommendedEnv}, script=${sweepBestRunScript}`
      : '⏭️ skipped';
    const benchmarkScorecardStatus = benchmarkScorecardEnabled
      ? `overall=${benchmarkOverallStatus}, hybrid=${benchmarkHybridStatus}, graph2d=${benchmarkGraph2dStatus}, history=${benchmarkHistoryStatus}, brep=${benchmarkBrepStatus}, governance=${benchmarkGovernanceStatus}, assistant=${benchmarkAssistantStatus}, review_queue=${benchmarkReviewQueueStatus}, feedback_flywheel=${benchmarkFeedbackFlywheelStatus}, ocr=${benchmarkOcrStatus}, qdrant=${benchmarkQdrantStatus}, knowledge=${benchmarkKnowledgeStatus}, engineering=${benchmarkEngineeringStatus}, operator_adoption=${benchmarkScorecardOperatorAdoptionStatus || 'n/a'}, operator_mode=${benchmarkScorecardOperatorAdoptionMode || 'n/a'}, operator_outcome_drift=${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'}, operator_outcome_drift_summary=${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkEngineeringStatusLine = benchmarkEngineeringEnabled
      ? `status=${benchmarkEngineeringArtifactStatus}, coverage=${benchmarkEngineeringArtifactCoverageRatio}, violations=${benchmarkEngineeringRowsWithViolations}, standards=${benchmarkEngineeringRowsWithStandards}, ocr_standards=${benchmarkEngineeringOcrStandards}, artifact=${benchmarkEngineeringArtifact}`
      : '⏭️ skipped';
    const benchmarkRealdataStatusLine = benchmarkRealdataEnabled
      ? `status=${benchmarkRealdataStatus}, ready=${benchmarkRealdataReadyComponents}, partial=${benchmarkRealdataPartialComponents}, blocked=${benchmarkRealdataEnvironmentBlocked}, available=${benchmarkRealdataAvailableComponents}, hybrid=${benchmarkRealdataHybridStatus}, history=${benchmarkRealdataHistoryStatus}, step_smoke=${benchmarkRealdataStepSmokeStatus}, step_dir=${benchmarkRealdataStepDirStatus}, recommendations=${benchmarkRealdataRecommendations || 'n/a'}, artifact=${benchmarkRealdataArtifact}`
      : '⏭️ skipped';
    const benchmarkRealdataScorecardStatusLine = benchmarkRealdataScorecardEnabled
      ? `status=${benchmarkRealdataScorecardStatus}, ready=${benchmarkRealdataScorecardReadyComponents}, partial=${benchmarkRealdataScorecardPartialComponents}, blocked=${benchmarkRealdataScorecardEnvironmentBlocked}, available=${benchmarkRealdataScorecardAvailableComponents}, best_surface=${benchmarkRealdataScorecardBestSurface || 'n/a'}, hybrid=${benchmarkRealdataScorecardHybridStatus}, history=${benchmarkRealdataScorecardHistoryStatus}, step_smoke=${benchmarkRealdataScorecardStepSmokeStatus}, step_dir=${benchmarkRealdataScorecardStepDirStatus}, recommendations=${benchmarkRealdataScorecardRecommendations || 'n/a'}, artifact=${benchmarkRealdataScorecardArtifact}`
      : '⏭️ skipped';
    const benchmarkCompetitiveSurpassStatusLine = benchmarkCompetitiveSurpassEnabled
      ? `status=${benchmarkCompetitiveSurpassStatus}, score=${benchmarkCompetitiveSurpassScore}, ready=${benchmarkCompetitiveSurpassReadyPillars || 'n/a'}, partial=${benchmarkCompetitiveSurpassPartialPillars || 'n/a'}, blocked=${benchmarkCompetitiveSurpassBlockedPillars || 'n/a'}, gaps=${benchmarkCompetitiveSurpassPrimaryGaps || 'n/a'}, recommendations=${benchmarkCompetitiveSurpassRecommendations || 'n/a'}, artifact=${benchmarkCompetitiveSurpassArtifact}`
      : '⏭️ skipped';
    const benchmarkCompetitiveSurpassTrendStatusLine = benchmarkCompetitiveSurpassTrendEnabled
      ? `status=${benchmarkCompetitiveSurpassTrendStatus || 'n/a'}, score_delta=${benchmarkCompetitiveSurpassTrendScoreDelta || 0}, improved=${benchmarkCompetitiveSurpassTrendPillarImprovements || 'n/a'}, regressed=${benchmarkCompetitiveSurpassTrendPillarRegressions || 'n/a'}, resolved_gaps=${benchmarkCompetitiveSurpassTrendResolvedPrimaryGaps || 'n/a'}, new_gaps=${benchmarkCompetitiveSurpassTrendNewPrimaryGaps || 'n/a'}, recommendations=${benchmarkCompetitiveSurpassTrendRecommendations || 'n/a'}, artifact=${benchmarkCompetitiveSurpassTrendArtifact}`
      : '⏭️ skipped';
    const benchmarkCompetitiveSurpassActionPlanStatusLine = benchmarkCompetitiveSurpassActionPlanEnabled
      ? `status=${benchmarkCompetitiveSurpassActionPlanStatus || 'n/a'}, actions=${benchmarkCompetitiveSurpassActionPlanTotalActionCount || 'n/a'}, high=${benchmarkCompetitiveSurpassActionPlanHighPriorityActionCount || 'n/a'}, medium=${benchmarkCompetitiveSurpassActionPlanMediumPriorityActionCount || 'n/a'}, pillars=${benchmarkCompetitiveSurpassActionPlanPriorityPillars || 'n/a'}, first_actions=${benchmarkCompetitiveSurpassActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkCompetitiveSurpassActionPlanRecommendations || 'n/a'}, artifact=${benchmarkCompetitiveSurpassActionPlanArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeStatusLine = benchmarkKnowledgeEnabled
      ? `status=${benchmarkKnowledgeArtifactStatus}, refs=${benchmarkKnowledgeTotalReferenceItems}, ready=${benchmarkKnowledgeReadyComponents}, partial=${benchmarkKnowledgePartialComponents}, missing=${benchmarkKnowledgeMissingComponents}, domain_count=${benchmarkKnowledgeDomainCount}, domains=${benchmarkKnowledgePriorityDomains || 'n/a'}, domain_focus=${benchmarkKnowledgeDomainFocusAreas || 'n/a'}, focus_count=${benchmarkKnowledgeFocusAreaCount}, focus=${benchmarkKnowledgeFocusAreas || benchmarkKnowledgeFocusAreasScorecard || 'n/a'}, artifact=${benchmarkKnowledgeArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDriftStatusLine = benchmarkKnowledgeDriftEnabled
      ? `status=${benchmarkKnowledgeDriftStatus}, current=${benchmarkKnowledgeDriftCurrentStatus}, previous=${benchmarkKnowledgeDriftPreviousStatus}, delta=${benchmarkKnowledgeDriftReferenceItemDelta}, regressions=${benchmarkKnowledgeDriftRegressions || 'n/a'}, improvements=${benchmarkKnowledgeDriftImprovements || 'n/a'}, domain_regressions=${benchmarkKnowledgeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkKnowledgeDriftDomainImprovements || 'n/a'}, resolved=${benchmarkKnowledgeDriftResolvedFocusAreas || 'n/a'}, new=${benchmarkKnowledgeDriftNewFocusAreas || 'n/a'}, resolved_domains=${benchmarkKnowledgeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkKnowledgeDriftNewPriorityDomains || 'n/a'}, artifact=${benchmarkKnowledgeDriftArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeApplicationStatusLine = benchmarkKnowledgeApplicationEnabled
      ? `status=${benchmarkKnowledgeApplicationStatus}, ready=${benchmarkKnowledgeApplicationReadyDomains}, partial=${benchmarkKnowledgeApplicationPartialDomains}, missing=${benchmarkKnowledgeApplicationMissingDomains}, total=${benchmarkKnowledgeApplicationTotalDomains}, focus_count=${benchmarkKnowledgeApplicationFocusAreaCount}, focus=${benchmarkKnowledgeApplicationFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeApplicationPriorityDomains || 'n/a'} / ${benchmarkKnowledgeApplicationDomainStatuses || 'n/a'}, recommendations=${benchmarkKnowledgeApplicationRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeApplicationArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeRealdataCorrelationStatusLine = benchmarkKnowledgeRealdataCorrelationEnabled
      ? `status=${benchmarkKnowledgeRealdataCorrelationStatus}, ready=${benchmarkKnowledgeRealdataCorrelationReadyDomains}, partial=${benchmarkKnowledgeRealdataCorrelationPartialDomains}, blocked=${benchmarkKnowledgeRealdataCorrelationBlockedDomains}, total=${benchmarkKnowledgeRealdataCorrelationTotalDomains}, focus=${benchmarkKnowledgeRealdataCorrelationFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeRealdataCorrelationPriorityDomains || 'n/a'} / ${benchmarkKnowledgeRealdataCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkKnowledgeRealdataCorrelationRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeRealdataCorrelationArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainMatrixStatusLine = benchmarkKnowledgeDomainMatrixEnabled
      ? `status=${benchmarkKnowledgeDomainMatrixStatus}, ready=${benchmarkKnowledgeDomainMatrixReadyDomains}, partial=${benchmarkKnowledgeDomainMatrixPartialDomains}, blocked=${benchmarkKnowledgeDomainMatrixBlockedDomains}, total=${benchmarkKnowledgeDomainMatrixTotalDomains}, focus=${benchmarkKnowledgeDomainMatrixFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeDomainMatrixPriorityDomains || 'n/a'} / ${benchmarkKnowledgeDomainMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkKnowledgeDomainMatrixRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainMatrixArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainCapabilityMatrixStatusLine = benchmarkKnowledgeDomainCapabilityMatrixEnabled
      ? `status=${benchmarkKnowledgeDomainCapabilityMatrixStatus}, ready=${benchmarkKnowledgeDomainCapabilityMatrixReadyDomains}, partial=${benchmarkKnowledgeDomainCapabilityMatrixPartialDomains}, blocked=${benchmarkKnowledgeDomainCapabilityMatrixBlockedDomains}, total=${benchmarkKnowledgeDomainCapabilityMatrixTotalDomains}, focus=${benchmarkKnowledgeDomainCapabilityMatrixFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeDomainCapabilityMatrixPriorityDomains || 'n/a'} / ${benchmarkKnowledgeDomainCapabilityMatrixDomainStatuses || 'n/a'}, provider_gaps=${benchmarkKnowledgeDomainCapabilityMatrixProviderGapDomains || 'n/a'}, surface_gaps=${benchmarkKnowledgeDomainCapabilityMatrixSurfaceGapDomains || 'n/a'}, recommendations=${benchmarkKnowledgeDomainCapabilityMatrixRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainCapabilityMatrixArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainApiSurfaceMatrixStatusLine = benchmarkKnowledgeDomainApiSurfaceMatrixEnabled
      ? `status=${benchmarkKnowledgeDomainApiSurfaceMatrixStatus}, ready=${benchmarkKnowledgeDomainApiSurfaceMatrixReadyDomains}, partial=${benchmarkKnowledgeDomainApiSurfaceMatrixPartialDomains}, blocked=${benchmarkKnowledgeDomainApiSurfaceMatrixBlockedDomains}, total=${benchmarkKnowledgeDomainApiSurfaceMatrixTotalDomains}, routes=${benchmarkKnowledgeDomainApiSurfaceMatrixTotalApiRoutes || 'n/a'}, focus=${benchmarkKnowledgeDomainApiSurfaceMatrixFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeDomainApiSurfaceMatrixPriorityDomains || 'n/a'} / ${benchmarkKnowledgeDomainApiSurfaceMatrixDomainStatuses || 'n/a'}, public_api_gaps=${benchmarkKnowledgeDomainApiSurfaceMatrixPublicApiGapDomains || 'n/a'}, reference_gaps=${benchmarkKnowledgeDomainApiSurfaceMatrixReferenceGapDomains || 'n/a'}, recommendations=${benchmarkKnowledgeDomainApiSurfaceMatrixRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainApiSurfaceMatrixArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainCapabilityDriftStatusLine = benchmarkKnowledgeDomainCapabilityDriftEnabled
      ? `status=${benchmarkKnowledgeDomainCapabilityDriftStatus || 'n/a'}, current=${benchmarkKnowledgeDomainCapabilityDriftCurrentStatus || 'n/a'}, previous=${benchmarkKnowledgeDomainCapabilityDriftPreviousStatus || 'n/a'}, provider_gap_delta=${benchmarkKnowledgeDomainCapabilityDriftProviderGapDelta || 'n/a'}, surface_gap_delta=${benchmarkKnowledgeDomainCapabilityDriftSurfaceGapDelta || 'n/a'}, domain_regressions=${benchmarkKnowledgeDomainCapabilityDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkKnowledgeDomainCapabilityDriftDomainImprovements || 'n/a'}, recommendations=${benchmarkKnowledgeDomainCapabilityDriftRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainCapabilityDriftArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainActionPlanStatusLine = benchmarkKnowledgeDomainActionPlanEnabled
      ? `status=${benchmarkKnowledgeDomainActionPlanStatus || 'n/a'}, ready=${benchmarkKnowledgeDomainActionPlanReadyDomains || 'n/a'}, partial=${benchmarkKnowledgeDomainActionPlanPartialDomains || 'n/a'}, blocked=${benchmarkKnowledgeDomainActionPlanBlockedDomains || 'n/a'}, total=${benchmarkKnowledgeDomainActionPlanTotalDomains || 'n/a'}, actions=${benchmarkKnowledgeDomainActionPlanTotalActions || 'n/a'}, high=${benchmarkKnowledgeDomainActionPlanHighPriorityActions || 'n/a'}, medium=${benchmarkKnowledgeDomainActionPlanMediumPriorityActions || 'n/a'}, priority_domains=${benchmarkKnowledgeDomainActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkKnowledgeDomainActionPlanRecommendedFirstActions || 'n/a'}, domain_action_counts=${benchmarkKnowledgeDomainActionPlanDomainActionCounts || 'n/a'}, recommendations=${benchmarkKnowledgeDomainActionPlanRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainActionPlanArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainSurfaceActionPlanStatusLine = benchmarkKnowledgeDomainSurfaceActionPlanEnabled
      ? `status=${benchmarkKnowledgeDomainSurfaceActionPlanStatus || 'n/a'}, subcapabilities=${benchmarkKnowledgeDomainSurfaceActionPlanTotalSubcapabilityCount || 'n/a'}, actions=${benchmarkKnowledgeDomainSurfaceActionPlanTotalActionCount || 'n/a'}, high=${benchmarkKnowledgeDomainSurfaceActionPlanHighPriorityActionCount || 'n/a'}, medium=${benchmarkKnowledgeDomainSurfaceActionPlanMediumPriorityActionCount || 'n/a'}, priority_domains=${benchmarkKnowledgeDomainSurfaceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkKnowledgeDomainSurfaceActionPlanRecommendedFirstActions || 'n/a'}, domain_action_counts=${benchmarkKnowledgeDomainSurfaceActionPlanDomainActionCounts || 'n/a'}, recommendations=${benchmarkKnowledgeDomainSurfaceActionPlanRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainSurfaceActionPlanArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanStatusLine = benchmarkKnowledgeDomainReleaseReadinessActionPlanEnabled
      ? `status=${benchmarkKnowledgeDomainReleaseReadinessActionPlanStatus || 'n/a'}, actions=${benchmarkKnowledgeDomainReleaseReadinessActionPlanTotalActionCount || 'n/a'}, high=${benchmarkKnowledgeDomainReleaseReadinessActionPlanHighPriorityActionCount || 'n/a'}, medium=${benchmarkKnowledgeDomainReleaseReadinessActionPlanMediumPriorityActionCount || 'n/a'}, gate_open=${benchmarkKnowledgeDomainReleaseReadinessActionPlanGateOpen || 'n/a'}, priority_domains=${benchmarkKnowledgeDomainReleaseReadinessActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkKnowledgeDomainReleaseReadinessActionPlanRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainReleaseReadinessActionPlanArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainControlPlaneStatusLine = benchmarkKnowledgeDomainControlPlaneEnabled
      ? `status=${benchmarkKnowledgeDomainControlPlaneStatus || 'n/a'}, ready=${benchmarkKnowledgeDomainControlPlaneReadyDomains || 'n/a'}, partial=${benchmarkKnowledgeDomainControlPlanePartialDomains || 'n/a'}, blocked=${benchmarkKnowledgeDomainControlPlaneBlockedDomains || 'n/a'}, missing=${benchmarkKnowledgeDomainControlPlaneMissingDomains || 'n/a'}, total=${benchmarkKnowledgeDomainControlPlaneTotalDomains || 'n/a'}, actions=${benchmarkKnowledgeDomainControlPlaneTotalActions || 'n/a'}, high=${benchmarkKnowledgeDomainControlPlaneHighPriorityActions || 'n/a'}, release_blockers=${benchmarkKnowledgeDomainControlPlaneReleaseBlockers || 'n/a'}, priority_domains=${benchmarkKnowledgeDomainControlPlanePriorityDomains || 'n/a'}, focus_areas=${benchmarkKnowledgeDomainControlPlaneFocusAreas || 'n/a'}, recommendations=${benchmarkKnowledgeDomainControlPlaneRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainControlPlaneArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainControlPlaneDriftStatusLine = benchmarkKnowledgeDomainControlPlaneDriftEnabled
      ? `status=${benchmarkKnowledgeDomainControlPlaneDriftStatus || 'n/a'}, current=${benchmarkKnowledgeDomainControlPlaneDriftCurrentStatus || 'n/a'}, previous=${benchmarkKnowledgeDomainControlPlaneDriftPreviousStatus || 'n/a'}, ready_delta=${benchmarkKnowledgeDomainControlPlaneDriftReadyDomainDelta || 'n/a'}, blocked_delta=${benchmarkKnowledgeDomainControlPlaneDriftBlockedDomainDelta || 'n/a'}, total_action_delta=${benchmarkKnowledgeDomainControlPlaneDriftTotalActionDelta || 'n/a'}, high_priority_action_delta=${benchmarkKnowledgeDomainControlPlaneDriftHighPriorityActionDelta || 'n/a'}, domain_regressions=${benchmarkKnowledgeDomainControlPlaneDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkKnowledgeDomainControlPlaneDriftDomainImprovements || 'n/a'}, resolved_release_blockers=${benchmarkKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers || 'n/a'}, new_release_blockers=${benchmarkKnowledgeDomainControlPlaneDriftNewReleaseBlockers || 'n/a'}, recommendations=${benchmarkKnowledgeDomainControlPlaneDriftRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainControlPlaneDriftArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainReleaseGateStatusLine = benchmarkKnowledgeDomainReleaseGateEnabled
      ? `status=${benchmarkKnowledgeDomainReleaseGateStatus || 'n/a'}, gate_open=${benchmarkKnowledgeDomainReleaseGateGateOpen || 'n/a'}, releasable=${benchmarkKnowledgeDomainReleaseGateReleasableDomainCount || 'n/a'}, blocked=${benchmarkKnowledgeDomainReleaseGateBlockedDomainCount || 'n/a'}, partial=${benchmarkKnowledgeDomainReleaseGatePartialDomainCount || 'n/a'}, priority_domains=${benchmarkKnowledgeDomainReleaseGatePriorityDomains || 'n/a'}, releasable_domains=${benchmarkKnowledgeDomainReleaseGateReleasableDomains || 'n/a'}, blocked_domains=${benchmarkKnowledgeDomainReleaseGateBlockedDomains || 'n/a'}, blocking_reasons=${benchmarkKnowledgeDomainReleaseGateBlockingReasons || 'n/a'}, warnings=${benchmarkKnowledgeDomainReleaseGateWarningReasons || 'n/a'}, first_action=${benchmarkKnowledgeDomainReleaseGateRecommendedFirstAction || 'n/a'}, recommendations=${benchmarkKnowledgeDomainReleaseGateRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeDomainReleaseGateArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatusLine = benchmarkKnowledgeDomainReleaseSurfaceAlignmentEnabled
      ? `status=${benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatus || 'n/a'}, summary=${benchmarkKnowledgeDomainReleaseSurfaceAlignmentSummary || 'n/a'}, mismatches=${benchmarkKnowledgeDomainReleaseSurfaceAlignmentMismatches || 'n/a'}, artifact=${benchmarkKnowledgeDomainReleaseSurfaceAlignmentArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeReferenceInventoryStatusLine = benchmarkKnowledgeReferenceInventoryEnabled
      ? `status=${benchmarkKnowledgeReferenceInventoryStatus || 'n/a'}, ready=${benchmarkKnowledgeReferenceInventoryReadyDomains || 'n/a'}, partial=${benchmarkKnowledgeReferenceInventoryPartialDomains || 'n/a'}, blocked=${benchmarkKnowledgeReferenceInventoryBlockedDomains || 'n/a'}, total=${benchmarkKnowledgeReferenceInventoryTotalDomains || 'n/a'}, refs=${benchmarkKnowledgeReferenceInventoryTotalReferenceItems || 'n/a'}, tables=${benchmarkKnowledgeReferenceInventoryPopulatedTables || 'n/a'}/${benchmarkKnowledgeReferenceInventoryTotalTables || 'n/a'}, domains=${benchmarkKnowledgeReferenceInventoryPriorityDomains || 'n/a'}, focus_tables=${benchmarkKnowledgeReferenceInventoryFocusTables || 'n/a'}, recommendations=${benchmarkKnowledgeReferenceInventoryRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeReferenceInventoryArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeSourceCoverageStatusLine = benchmarkKnowledgeSourceCoverageEnabled
      ? `status=${benchmarkKnowledgeSourceCoverageStatus || 'n/a'}, ready=${benchmarkKnowledgeSourceCoverageReadySourceGroups || 'n/a'}, partial=${benchmarkKnowledgeSourceCoveragePartialSourceGroups || 'n/a'}, missing=${benchmarkKnowledgeSourceCoverageMissingSourceGroups || 'n/a'}, total=${benchmarkKnowledgeSourceCoverageTotalSourceGroups || 'n/a'}, tables=${benchmarkKnowledgeSourceCoverageTotalSourceTables || 'n/a'}, items=${benchmarkKnowledgeSourceCoverageTotalSourceItems || 'n/a'}, standards=${benchmarkKnowledgeSourceCoverageTotalReferenceStandards || 'n/a'}, expansion_ready=${benchmarkKnowledgeSourceCoverageReadyExpansionCandidates || 'n/a'}, focus=${benchmarkKnowledgeSourceCoverageFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeSourceCoveragePriorityDomains || 'n/a'} / ${benchmarkKnowledgeSourceCoverageDomainStatuses || 'n/a'}, expansion_candidates=${benchmarkKnowledgeSourceCoverageExpansionCandidates || 'n/a'}, recommendations=${benchmarkKnowledgeSourceCoverageRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeSourceCoverageArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeSourceActionPlanStatusLine = benchmarkKnowledgeSourceActionPlanEnabled
      ? `status=${benchmarkKnowledgeSourceActionPlanStatus || 'n/a'}, actions=${benchmarkKnowledgeSourceActionPlanTotalActions || 'n/a'}, high=${benchmarkKnowledgeSourceActionPlanHighPriorityActions || 'n/a'}, medium=${benchmarkKnowledgeSourceActionPlanMediumPriorityActions || 'n/a'}, expansion=${benchmarkKnowledgeSourceActionPlanExpansionActions || 'n/a'}, priority_domains=${benchmarkKnowledgeSourceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkKnowledgeSourceActionPlanRecommendedFirstActions || 'n/a'}, source_group_action_counts=${benchmarkKnowledgeSourceActionPlanSourceGroupActionCounts || 'n/a'}, recommendations=${benchmarkKnowledgeSourceActionPlanRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeSourceActionPlanArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeSourceDriftStatusLine = benchmarkKnowledgeSourceDriftEnabled
      ? `status=${benchmarkKnowledgeSourceDriftStatus || 'n/a'}, current=${benchmarkKnowledgeSourceDriftCurrentStatus || 'n/a'}, previous=${benchmarkKnowledgeSourceDriftPreviousStatus || 'n/a'}, ready_delta=${benchmarkKnowledgeSourceDriftReadySourceGroupDelta || 'n/a'}, missing_delta=${benchmarkKnowledgeSourceDriftMissingSourceGroupDelta || 'n/a'}, regressions=${benchmarkKnowledgeSourceDriftRegressions || 'n/a'}, improvements=${benchmarkKnowledgeSourceDriftImprovements || 'n/a'}, source_group_regressions=${benchmarkKnowledgeSourceDriftSourceGroupRegressions || 'n/a'}, source_group_improvements=${benchmarkKnowledgeSourceDriftSourceGroupImprovements || 'n/a'}, resolved_focus=${benchmarkKnowledgeSourceDriftResolvedFocusAreas || 'n/a'}, new_focus=${benchmarkKnowledgeSourceDriftNewFocusAreas || 'n/a'}, resolved_domains=${benchmarkKnowledgeSourceDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkKnowledgeSourceDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkKnowledgeSourceDriftRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeSourceDriftArtifact || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkKnowledgeOutcomeCorrelationStatusLine = benchmarkKnowledgeOutcomeCorrelationEnabled
      ? `status=${benchmarkKnowledgeOutcomeCorrelationStatus}, ready=${benchmarkKnowledgeOutcomeCorrelationReadyDomains}, partial=${benchmarkKnowledgeOutcomeCorrelationPartialDomains}, blocked=${benchmarkKnowledgeOutcomeCorrelationBlockedDomains}, total=${benchmarkKnowledgeOutcomeCorrelationTotalDomains}, focus=${benchmarkKnowledgeOutcomeCorrelationFocusAreas || 'n/a'}, domains=${benchmarkKnowledgeOutcomeCorrelationPriorityDomains || 'n/a'} / ${benchmarkKnowledgeOutcomeCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkKnowledgeOutcomeCorrelationRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeOutcomeCorrelationArtifact}`
      : '⏭️ skipped';
    const benchmarkKnowledgeOutcomeDriftStatusLine = benchmarkKnowledgeOutcomeDriftEnabled
      ? `status=${benchmarkKnowledgeOutcomeDriftStatus || 'n/a'}, current=${benchmarkKnowledgeOutcomeDriftCurrentStatus || 'n/a'}, previous=${benchmarkKnowledgeOutcomeDriftPreviousStatus || 'n/a'}, ready_delta=${benchmarkKnowledgeOutcomeDriftReadyDomainDelta || 'n/a'}, blocked_delta=${benchmarkKnowledgeOutcomeDriftBlockedDomainDelta || 'n/a'}, regressions=${benchmarkKnowledgeOutcomeDriftRegressions || 'n/a'}, improvements=${benchmarkKnowledgeOutcomeDriftImprovements || 'n/a'}, domain_regressions=${benchmarkKnowledgeOutcomeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkKnowledgeOutcomeDriftDomainImprovements || 'n/a'}, resolved_focus=${benchmarkKnowledgeOutcomeDriftResolvedFocusAreas || 'n/a'}, new_focus=${benchmarkKnowledgeOutcomeDriftNewFocusAreas || 'n/a'}, resolved_domains=${benchmarkKnowledgeOutcomeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkKnowledgeOutcomeDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkKnowledgeOutcomeDriftRecommendations || 'n/a'}, artifact=${benchmarkKnowledgeOutcomeDriftArtifact}`
      : '⏭️ skipped';
    const feedbackFlywheelBenchmarkStatusLine = feedbackFlywheelBenchmarkEnabled
      ? `status=${feedbackFlywheelBenchmarkStatus}, feedback=${feedbackFlywheelBenchmarkTotal}, corrections=${feedbackFlywheelBenchmarkCorrections}, finetune=${feedbackFlywheelBenchmarkFinetuneSamples}, triplets=${feedbackFlywheelBenchmarkMetricTriplets}, artifact=${feedbackFlywheelBenchmarkArtifact}`
      : '⏭️ skipped';
    const benchmarkOperationalSummaryStatus = benchmarkOperationalSummaryEnabled
      ? `overall=${benchmarkOperationalSummaryOverall}, feedback=${benchmarkOperationalFeedbackStatus}, assistant=${benchmarkOperationalAssistantStatus}, review_queue=${benchmarkOperationalReviewQueueStatus}, ocr=${benchmarkOperationalOcrStatus}, operator_adoption=${benchmarkOperationalOperatorAdoptionStatus || 'n/a'}, operator_outcome_drift=${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'}, operator_outcome_drift_summary=${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'}, blockers=${benchmarkOperationalBlockers || 'n/a'}, recommendations=${benchmarkOperationalRecommendations || 'n/a'}, artifact=${benchmarkOperationalArtifact}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleStatus = benchmarkArtifactBundleEnabled
      ? `overall=${benchmarkArtifactBundleOverall}, available_artifacts=${benchmarkArtifactBundleAvailableArtifacts}, feedback=${benchmarkArtifactBundleFeedbackStatus}, assistant=${benchmarkArtifactBundleAssistantStatus}, review_queue=${benchmarkArtifactBundleReviewQueueStatus}, ocr=${benchmarkArtifactBundleOcrStatus}, knowledge=${benchmarkArtifactBundleKnowledgeStatus}, knowledge_drift=${benchmarkArtifactBundleKnowledgeDriftStatus}, knowledge_drift_summary=${benchmarkArtifactBundleKnowledgeDriftSummary || 'n/a'}, knowledge_drift_changes=${benchmarkArtifactBundleKnowledgeDriftChanges || 'n/a'}, knowledge_drift_domain_regressions=${benchmarkArtifactBundleKnowledgeDriftDomainRegressions || 'n/a'}, knowledge_drift_domain_improvements=${benchmarkArtifactBundleKnowledgeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkArtifactBundleKnowledgeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkArtifactBundleKnowledgeDriftNewPriorityDomains || 'n/a'}, knowledge_domains=${benchmarkArtifactBundleKnowledgePriorityDomains || 'n/a'}, knowledge_domain_focus=${benchmarkArtifactBundleKnowledgeDomainFocusAreas || 'n/a'}, knowledge_focus=${benchmarkArtifactBundleKnowledgeFocusAreas || 'n/a'}, engineering=${benchmarkArtifactBundleEngineeringStatus}, realdata=${benchmarkArtifactBundleRealdataStatus || 'n/a'}, realdata_recommendations=${benchmarkArtifactBundleRealdataRecommendations || 'n/a'}, operator_drift=${benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftStatus}, operator_drift_summary=${benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftSummary || 'n/a'}, blockers=${benchmarkArtifactBundleBlockers || 'n/a'}, recommendations=${benchmarkArtifactBundleRecommendations || 'n/a'}, artifact=${benchmarkArtifactBundleArtifact}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeApplicationStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeApplicationStatus || 'n/a'}, focus=${benchmarkArtifactBundleKnowledgeApplicationFocusAreas || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeApplicationPriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeApplicationDomainStatuses || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeApplicationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeRealdataCorrelationStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeRealdataCorrelationStatus || 'n/a'}, focus=${benchmarkArtifactBundleKnowledgeRealdataCorrelationFocusAreas || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeRealdataCorrelationPriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeRealdataCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeRealdataCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainMatrixStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainMatrixStatus || 'n/a'}, focus=${benchmarkArtifactBundleKnowledgeDomainMatrixFocusAreas || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeDomainMatrixPriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeDomainMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixStatus || 'n/a'}, focus=${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixFocusAreas || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixPriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainCapabilityDriftStatus = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityDriftDomainRegressions = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityDriftDomainImprovements = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityDriftRecommendations = envStr("STEP_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS", "");
    const benchmarkArtifactBundleKnowledgeDomainCapabilityDriftStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainCapabilityDriftStatus || 'n/a'}, regressions=${benchmarkArtifactBundleKnowledgeDomainCapabilityDriftDomainRegressions || 'n/a'}, improvements=${benchmarkArtifactBundleKnowledgeDomainCapabilityDriftDomainImprovements || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainCapabilityDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainActionPlanStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainActionPlanStatus || 'n/a'}, actions=${benchmarkArtifactBundleKnowledgeDomainActionPlanActions || 'n/a'}, priority_domains=${benchmarkArtifactBundleKnowledgeDomainActionPlanPriorityDomains || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanStatus || 'n/a'}, actions=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanTotalActionCount || 'n/a'}, gate_open=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanGateOpen || 'n/a'}, priority_domains=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainControlPlaneStatus || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDomains || 'n/a'}, release_blockers=${benchmarkArtifactBundleKnowledgeDomainControlPlaneReleaseBlockers || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainControlPlaneRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftStatus || 'n/a'}, regressions=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftDomainRegressions || 'n/a'}, improvements=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftDomainImprovements || 'n/a'}, resolved_release_blockers=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers || 'n/a'}, new_release_blockers=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftNewReleaseBlockers || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeDomainReleaseGateStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeDomainReleaseGateStatus || 'n/a'}, summary=${benchmarkArtifactBundleKnowledgeDomainReleaseGateSummary || 'n/a'}, gate_open=${benchmarkArtifactBundleKnowledgeDomainReleaseGateGateOpen || 'n/a'}, releasable_domains=${benchmarkArtifactBundleKnowledgeDomainReleaseGateReleasableDomains || 'n/a'}, blocked_domains=${benchmarkArtifactBundleKnowledgeDomainReleaseGateBlockedDomains || 'n/a'}, priority_domains=${benchmarkArtifactBundleKnowledgeDomainReleaseGatePriorityDomains || 'n/a'}, blocking_reasons=${benchmarkArtifactBundleKnowledgeDomainReleaseGateBlockingReasons || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeDomainReleaseGateRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeReferenceInventoryStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeReferenceInventoryStatus || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeReferenceInventoryPriorityDomains || 'n/a'}, total_items=${benchmarkArtifactBundleKnowledgeReferenceInventoryTotalReferenceItems || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeSourceCoverageStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeSourceCoverageStatus || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeSourceCoverageDomainStatuses || 'n/a'}, expansion_candidates=${benchmarkArtifactBundleKnowledgeSourceCoverageExpansionCandidates || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeSourceCoverageRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeSourceActionPlanStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeSourceActionPlanStatus || 'n/a'}, priority_domains=${benchmarkArtifactBundleKnowledgeSourceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkArtifactBundleKnowledgeSourceActionPlanRecommendedFirstActions || 'n/a'}, source_group_action_counts=${benchmarkArtifactBundleKnowledgeSourceActionPlanSourceGroupActionCounts || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeSourceActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeSourceDriftStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeSourceDriftStatus || 'n/a'}, summary=${benchmarkArtifactBundleKnowledgeSourceDriftSummary || 'n/a'}, source_group_regressions=${benchmarkArtifactBundleKnowledgeSourceDriftSourceGroupRegressions || 'n/a'}, source_group_improvements=${benchmarkArtifactBundleKnowledgeSourceDriftSourceGroupImprovements || 'n/a'}, resolved_domains=${benchmarkArtifactBundleKnowledgeSourceDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkArtifactBundleKnowledgeSourceDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeSourceDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatus || 'n/a'}, focus=${benchmarkArtifactBundleKnowledgeOutcomeCorrelationFocusAreas || 'n/a'}, domains=${benchmarkArtifactBundleKnowledgeOutcomeCorrelationPriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeOutcomeCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeOutcomeCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleKnowledgeOutcomeDriftStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleKnowledgeOutcomeDriftStatus || 'n/a'}, summary=${benchmarkArtifactBundleKnowledgeOutcomeDriftSummary || 'n/a'}, domain_regressions=${benchmarkArtifactBundleKnowledgeOutcomeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkArtifactBundleKnowledgeOutcomeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkArtifactBundleKnowledgeOutcomeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkArtifactBundleKnowledgeOutcomeDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkArtifactBundleKnowledgeOutcomeDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleCompetitiveSurpassStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleCompetitiveSurpassStatus || 'n/a'}, gaps=${benchmarkArtifactBundleCompetitiveSurpassPrimaryGaps || 'n/a'}, recommendations=${benchmarkArtifactBundleCompetitiveSurpassRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleCompetitiveSurpassTrendStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleCompetitiveSurpassTrendStatus || 'n/a'}, summary=${benchmarkArtifactBundleCompetitiveSurpassTrendSummary || 'n/a'}, recommendations=${benchmarkArtifactBundleCompetitiveSurpassTrendRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanStatusLine = benchmarkArtifactBundleEnabled
      ? `status=${benchmarkArtifactBundleCompetitiveSurpassActionPlanStatus || 'n/a'}, actions=${benchmarkArtifactBundleCompetitiveSurpassActionPlanTotalActionCount || 'n/a'}, pillars=${benchmarkArtifactBundleCompetitiveSurpassActionPlanPriorityPillars || 'n/a'}, recommendations=${benchmarkArtifactBundleCompetitiveSurpassActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionSummaryStatus = benchmarkCompanionSummaryEnabled
      ? `overall=${benchmarkCompanionSummaryOverall}, review_surface=${benchmarkCompanionReviewSurface}, primary_gap=${benchmarkCompanionPrimaryGap || 'n/a'}, hybrid=${benchmarkCompanionHybridStatus}, assistant=${benchmarkCompanionAssistantStatus}, review_queue=${benchmarkCompanionReviewQueueStatus}, ocr=${benchmarkCompanionOcrStatus}, qdrant=${benchmarkCompanionQdrantStatus}, knowledge=${benchmarkCompanionKnowledgeStatus}, knowledge_drift=${benchmarkCompanionKnowledgeDriftStatus}, knowledge_drift_summary=${benchmarkCompanionKnowledgeDriftSummary || 'n/a'}, knowledge_drift_changes=${benchmarkCompanionKnowledgeDriftChanges || 'n/a'}, knowledge_drift_domain_regressions=${benchmarkCompanionKnowledgeDriftDomainRegressions || 'n/a'}, knowledge_drift_domain_improvements=${benchmarkCompanionKnowledgeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkCompanionKnowledgeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkCompanionKnowledgeDriftNewPriorityDomains || 'n/a'}, knowledge_domains=${benchmarkCompanionKnowledgePriorityDomains || 'n/a'}, knowledge_domain_focus=${benchmarkCompanionKnowledgeDomainFocusAreas || 'n/a'}, knowledge_focus=${benchmarkCompanionKnowledgeFocusAreas || 'n/a'}, engineering=${benchmarkCompanionEngineeringStatus}, realdata=${benchmarkCompanionRealdataStatus || 'n/a'}, realdata_recommendations=${benchmarkCompanionRealdataRecommendations || 'n/a'}, operator_drift=${benchmarkCompanionOperatorAdoptionKnowledgeDriftStatus}, operator_drift_summary=${benchmarkCompanionOperatorAdoptionKnowledgeDriftSummary || 'n/a'}, blockers=${benchmarkCompanionBlockers || 'n/a'}, actions=${benchmarkCompanionRecommendedActions || 'n/a'}, artifact=${benchmarkCompanionArtifact}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeApplicationStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeApplicationStatus || 'n/a'}, focus=${benchmarkCompanionKnowledgeApplicationFocusAreas || 'n/a'}, domains=${benchmarkCompanionKnowledgeApplicationPriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeApplicationDomainStatuses || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeApplicationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeRealdataCorrelationStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeRealdataCorrelationStatus || 'n/a'}, focus=${benchmarkCompanionKnowledgeRealdataCorrelationFocusAreas || 'n/a'}, domains=${benchmarkCompanionKnowledgeRealdataCorrelationPriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeRealdataCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeRealdataCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainMatrixStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainMatrixStatus || 'n/a'}, focus=${benchmarkCompanionKnowledgeDomainMatrixFocusAreas || 'n/a'}, domains=${benchmarkCompanionKnowledgeDomainMatrixPriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeDomainMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainCapabilityMatrixStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainCapabilityMatrixStatus || 'n/a'}, focus=${benchmarkCompanionKnowledgeDomainCapabilityMatrixFocusAreas || 'n/a'}, domains=${benchmarkCompanionKnowledgeDomainCapabilityMatrixPriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeDomainCapabilityMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainCapabilityMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainCapabilityDriftStatus = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityDriftDomainRegressions = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityDriftDomainImprovements = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityDriftRecommendations = envStr("STEP_BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS", "");
    const benchmarkCompanionKnowledgeDomainCapabilityDriftStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainCapabilityDriftStatus || 'n/a'}, regressions=${benchmarkCompanionKnowledgeDomainCapabilityDriftDomainRegressions || 'n/a'}, improvements=${benchmarkCompanionKnowledgeDomainCapabilityDriftDomainImprovements || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainCapabilityDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainActionPlanStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainActionPlanStatus || 'n/a'}, actions=${benchmarkCompanionKnowledgeDomainActionPlanActions || 'n/a'}, priority_domains=${benchmarkCompanionKnowledgeDomainActionPlanPriorityDomains || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanStatus || 'n/a'}, actions=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanTotalActionCount || 'n/a'}, gate_open=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanGateOpen || 'n/a'}, priority_domains=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainControlPlaneStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainControlPlaneStatus || 'n/a'}, domains=${benchmarkCompanionKnowledgeDomainControlPlaneDomains || 'n/a'}, release_blockers=${benchmarkCompanionKnowledgeDomainControlPlaneReleaseBlockers || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainControlPlaneRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainControlPlaneDriftStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainControlPlaneDriftStatus || 'n/a'}, regressions=${benchmarkCompanionKnowledgeDomainControlPlaneDriftDomainRegressions || 'n/a'}, improvements=${benchmarkCompanionKnowledgeDomainControlPlaneDriftDomainImprovements || 'n/a'}, resolved_release_blockers=${benchmarkCompanionKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers || 'n/a'}, new_release_blockers=${benchmarkCompanionKnowledgeDomainControlPlaneDriftNewReleaseBlockers || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainControlPlaneDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeDomainReleaseGateStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeDomainReleaseGateStatus || 'n/a'}, summary=${benchmarkCompanionKnowledgeDomainReleaseGateSummary || 'n/a'}, gate_open=${benchmarkCompanionKnowledgeDomainReleaseGateGateOpen || 'n/a'}, releasable_domains=${benchmarkCompanionKnowledgeDomainReleaseGateReleasableDomains || 'n/a'}, blocked_domains=${benchmarkCompanionKnowledgeDomainReleaseGateBlockedDomains || 'n/a'}, priority_domains=${benchmarkCompanionKnowledgeDomainReleaseGatePriorityDomains || 'n/a'}, blocking_reasons=${benchmarkCompanionKnowledgeDomainReleaseGateBlockingReasons || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeDomainReleaseGateRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeReferenceInventoryStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeReferenceInventoryStatus || 'n/a'}, domains=${benchmarkCompanionKnowledgeReferenceInventoryPriorityDomains || 'n/a'}, total_items=${benchmarkCompanionKnowledgeReferenceInventoryTotalReferenceItems || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeSourceCoverageStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeSourceCoverageStatus || 'n/a'}, domains=${benchmarkCompanionKnowledgeSourceCoverageDomainStatuses || 'n/a'}, expansion_candidates=${benchmarkCompanionKnowledgeSourceCoverageExpansionCandidates || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeSourceCoverageRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeSourceActionPlanStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeSourceActionPlanStatus || 'n/a'}, priority_domains=${benchmarkCompanionKnowledgeSourceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkCompanionKnowledgeSourceActionPlanRecommendedFirstActions || 'n/a'}, source_group_action_counts=${benchmarkCompanionKnowledgeSourceActionPlanSourceGroupActionCounts || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeSourceActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeSourceDriftStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeSourceDriftStatus || 'n/a'}, summary=${benchmarkCompanionKnowledgeSourceDriftSummary || 'n/a'}, source_group_regressions=${benchmarkCompanionKnowledgeSourceDriftSourceGroupRegressions || 'n/a'}, source_group_improvements=${benchmarkCompanionKnowledgeSourceDriftSourceGroupImprovements || 'n/a'}, resolved_domains=${benchmarkCompanionKnowledgeSourceDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkCompanionKnowledgeSourceDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeSourceDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeOutcomeCorrelationStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeOutcomeCorrelationStatus || 'n/a'}, focus=${benchmarkCompanionKnowledgeOutcomeCorrelationFocusAreas || 'n/a'}, domains=${benchmarkCompanionKnowledgeOutcomeCorrelationPriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeOutcomeCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeOutcomeCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionKnowledgeOutcomeDriftStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionKnowledgeOutcomeDriftStatus || 'n/a'}, summary=${benchmarkCompanionKnowledgeOutcomeDriftSummary || 'n/a'}, domain_regressions=${benchmarkCompanionKnowledgeOutcomeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkCompanionKnowledgeOutcomeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkCompanionKnowledgeOutcomeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkCompanionKnowledgeOutcomeDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkCompanionKnowledgeOutcomeDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionCompetitiveSurpassStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionCompetitiveSurpassStatus || 'n/a'}, gaps=${benchmarkCompanionCompetitiveSurpassPrimaryGaps || 'n/a'}, recommendations=${benchmarkCompanionCompetitiveSurpassRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionCompetitiveSurpassTrendStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionCompetitiveSurpassTrendStatus || 'n/a'}, summary=${benchmarkCompanionCompetitiveSurpassTrendSummary || 'n/a'}, recommendations=${benchmarkCompanionCompetitiveSurpassTrendRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkCompanionCompetitiveSurpassActionPlanStatusLine = benchmarkCompanionSummaryEnabled
      ? `status=${benchmarkCompanionCompetitiveSurpassActionPlanStatus || 'n/a'}, actions=${benchmarkCompanionCompetitiveSurpassActionPlanTotalActionCount || 'n/a'}, pillars=${benchmarkCompanionCompetitiveSurpassActionPlanPriorityPillars || 'n/a'}, recommendations=${benchmarkCompanionCompetitiveSurpassActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseDecisionStatus = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseStatus}, automation_ready=${benchmarkReleaseAutomationReady}, source=${benchmarkReleasePrimarySignalSource}, hybrid=${benchmarkReleaseHybridStatus}, assistant=${benchmarkReleaseAssistantStatus}, review_queue=${benchmarkReleaseReviewQueueStatus}, ocr=${benchmarkReleaseOcrStatus}, qdrant=${benchmarkReleaseQdrantStatus}, knowledge=${benchmarkReleaseKnowledgeStatus}, knowledge_drift=${benchmarkReleaseKnowledgeDriftStatus}, knowledge_drift_summary=${benchmarkReleaseKnowledgeDriftSummary || 'n/a'}, knowledge_drift_domain_regressions=${benchmarkReleaseKnowledgeDriftDomainRegressions || 'n/a'}, knowledge_drift_domain_improvements=${benchmarkReleaseKnowledgeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseKnowledgeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseKnowledgeDriftNewPriorityDomains || 'n/a'}, knowledge_domains=${benchmarkReleaseKnowledgePriorityDomains || 'n/a'}, knowledge_domain_focus=${benchmarkReleaseKnowledgeDomainFocusAreas || 'n/a'}, knowledge_focus=${benchmarkReleaseKnowledgeFocusAreas || 'n/a'}, engineering=${benchmarkReleaseEngineeringStatus}, realdata=${benchmarkReleaseRealdataStatus || 'n/a'}, realdata_recommendations=${benchmarkReleaseRealdataRecommendations || 'n/a'}, operator_adoption=${benchmarkReleaseOperatorAdoptionStatus}, operator_drift=${benchmarkReleaseOperatorAdoptionKnowledgeDriftStatus}, operator_drift_summary=${benchmarkReleaseOperatorAdoptionKnowledgeDriftSummary || 'n/a'}, operator_outcome_drift=${benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'}, operator_outcome_drift_summary=${benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'}, blockers=${benchmarkReleaseBlockingSignals || 'n/a'}, review=${benchmarkReleaseReviewSignals || 'n/a'}, artifact=${benchmarkReleaseArtifact}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeApplicationStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeApplicationStatus || 'n/a'}, focus=${benchmarkReleaseKnowledgeApplicationFocusAreas || 'n/a'}, domains=${benchmarkReleaseKnowledgeApplicationPriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeApplicationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeApplicationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeRealdataCorrelationStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeRealdataCorrelationStatus || 'n/a'}, focus=${benchmarkReleaseKnowledgeRealdataCorrelationFocusAreas || 'n/a'}, domains=${benchmarkReleaseKnowledgeRealdataCorrelationPriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeRealdataCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeRealdataCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainMatrixStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainMatrixStatus || 'n/a'}, focus=${benchmarkReleaseKnowledgeDomainMatrixFocusAreas || 'n/a'}, domains=${benchmarkReleaseKnowledgeDomainMatrixPriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeDomainMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainCapabilityMatrixStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainCapabilityMatrixStatus || 'n/a'}, focus=${benchmarkReleaseKnowledgeDomainCapabilityMatrixFocusAreas || 'n/a'}, domains=${benchmarkReleaseKnowledgeDomainCapabilityMatrixPriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeDomainCapabilityMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainCapabilityMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainCapabilityDriftStatus = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseKnowledgeDomainCapabilityDriftStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainCapabilityDriftStatus || 'n/a'}, regressions=${benchmarkReleaseKnowledgeDomainCapabilityDriftDomainRegressions || 'n/a'}, improvements=${benchmarkReleaseKnowledgeDomainCapabilityDriftDomainImprovements || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainCapabilityDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainActionPlanStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseKnowledgeDomainActionPlanActions || 'n/a'}, priority_domains=${benchmarkReleaseKnowledgeDomainActionPlanPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanTotalActionCount || 'n/a'}, gate_open=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanGateOpen || 'n/a'}, priority_domains=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainControlPlaneStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainControlPlaneStatus || 'n/a'}, domains=${benchmarkReleaseKnowledgeDomainControlPlaneDomains || 'n/a'}, release_blockers=${benchmarkReleaseKnowledgeDomainControlPlaneReleaseBlockers || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainControlPlaneRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainControlPlaneDriftStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainControlPlaneDriftStatus || 'n/a'}, regressions=${benchmarkReleaseKnowledgeDomainControlPlaneDriftDomainRegressions || 'n/a'}, improvements=${benchmarkReleaseKnowledgeDomainControlPlaneDriftDomainImprovements || 'n/a'}, resolved_release_blockers=${benchmarkReleaseKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers || 'n/a'}, new_release_blockers=${benchmarkReleaseKnowledgeDomainControlPlaneDriftNewReleaseBlockers || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainControlPlaneDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeDomainReleaseGateStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeDomainReleaseGateStatus || 'n/a'}, gate_open=${benchmarkReleaseKnowledgeDomainReleaseGateGateOpen || 'n/a'}, releasable_domains=${benchmarkReleaseKnowledgeDomainReleaseGateReleasableDomains || 'n/a'}, blocked_domains=${benchmarkReleaseKnowledgeDomainReleaseGateBlockedDomains || 'n/a'}, priority_domains=${benchmarkReleaseKnowledgeDomainReleaseGatePriorityDomains || 'n/a'}, blocking_reasons=${benchmarkReleaseKnowledgeDomainReleaseGateBlockingReasons || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeDomainReleaseGateRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeReferenceInventoryStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeReferenceInventoryStatus || 'n/a'}, summary=${benchmarkReleaseKnowledgeReferenceInventorySummary || 'n/a'}, priority_domains=${benchmarkReleaseKnowledgeReferenceInventoryPriorityDomains || 'n/a'}, refs=${benchmarkReleaseKnowledgeReferenceInventoryTotalReferenceItems || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeReferenceInventoryRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeSourceCoverageStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeSourceCoverageStatus || 'n/a'}, domains=${benchmarkReleaseKnowledgeSourceCoverageDomainStatuses || 'n/a'}, expansion_candidates=${benchmarkReleaseKnowledgeSourceCoverageExpansionCandidates || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeSourceCoverageRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeSourceActionPlanStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeSourceActionPlanStatus || 'n/a'}, priority_domains=${benchmarkReleaseKnowledgeSourceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkReleaseKnowledgeSourceActionPlanRecommendedFirstActions || 'n/a'}, source_group_action_counts=${benchmarkReleaseKnowledgeSourceActionPlanSourceGroupActionCounts || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeSourceActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeSourceDriftStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeSourceDriftStatus || 'n/a'}, summary=${benchmarkReleaseKnowledgeSourceDriftSummary || 'n/a'}, source_group_regressions=${benchmarkReleaseKnowledgeSourceDriftSourceGroupRegressions || 'n/a'}, source_group_improvements=${benchmarkReleaseKnowledgeSourceDriftSourceGroupImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseKnowledgeSourceDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseKnowledgeSourceDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeSourceDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeOutcomeCorrelationStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeOutcomeCorrelationStatus || 'n/a'}, focus=${benchmarkReleaseKnowledgeOutcomeCorrelationFocusAreas || 'n/a'}, domains=${benchmarkReleaseKnowledgeOutcomeCorrelationPriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeOutcomeCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeOutcomeCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseKnowledgeOutcomeDriftStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseKnowledgeOutcomeDriftStatus || 'n/a'}, summary=${benchmarkReleaseKnowledgeOutcomeDriftSummary || 'n/a'}, domain_regressions=${benchmarkReleaseKnowledgeOutcomeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkReleaseKnowledgeOutcomeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseKnowledgeOutcomeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseKnowledgeOutcomeDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseKnowledgeOutcomeDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseCompetitiveSurpassStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseCompetitiveSurpassStatus || 'n/a'}, gaps=${benchmarkReleaseCompetitiveSurpassPrimaryGaps || 'n/a'}, recommendations=${benchmarkReleaseCompetitiveSurpassRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseCompetitiveSurpassTrendStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseCompetitiveSurpassTrendStatus || 'n/a'}, summary=${benchmarkReleaseCompetitiveSurpassTrendSummary || 'n/a'}, recommendations=${benchmarkReleaseCompetitiveSurpassTrendRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseCompetitiveSurpassActionPlanStatusLine = benchmarkReleaseDecisionEnabled
      ? `status=${benchmarkReleaseCompetitiveSurpassActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseCompetitiveSurpassActionPlanTotalActionCount || 'n/a'}, pillars=${benchmarkReleaseCompetitiveSurpassActionPlanPriorityPillars || 'n/a'}, recommendations=${benchmarkReleaseCompetitiveSurpassActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookStatus}, freeze_ready=${benchmarkReleaseRunbookFreezeReady}, source=${benchmarkReleaseRunbookPrimarySignalSource}, next=${benchmarkReleaseRunbookNextAction || 'n/a'}, knowledge=${benchmarkReleaseRunbookKnowledgeStatus}, knowledge_drift=${benchmarkReleaseRunbookKnowledgeDriftStatus}, knowledge_drift_summary=${benchmarkReleaseRunbookKnowledgeDriftSummary || 'n/a'}, knowledge_drift_domain_regressions=${benchmarkReleaseRunbookKnowledgeDriftDomainRegressions || 'n/a'}, knowledge_drift_domain_improvements=${benchmarkReleaseRunbookKnowledgeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseRunbookKnowledgeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseRunbookKnowledgeDriftNewPriorityDomains || 'n/a'}, knowledge_domains=${benchmarkReleaseRunbookKnowledgePriorityDomains || 'n/a'}, knowledge_domain_focus=${benchmarkReleaseRunbookKnowledgeDomainFocusAreas || 'n/a'}, knowledge_focus=${benchmarkReleaseRunbookKnowledgeFocusAreas || 'n/a'}, engineering=${benchmarkReleaseRunbookEngineeringStatus}, realdata=${benchmarkReleaseRunbookRealdataStatus || 'n/a'}, realdata_recommendations=${benchmarkReleaseRunbookRealdataRecommendations || 'n/a'}, operator_adoption=${benchmarkReleaseRunbookOperatorAdoptionStatus}, operator_drift=${benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftStatus}, operator_drift_summary=${benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftSummary || 'n/a'}, operator_outcome_drift=${benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'}, operator_outcome_drift_summary=${benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'}, missing=${benchmarkReleaseRunbookMissingArtifacts || 'n/a'}, blockers=${benchmarkReleaseRunbookBlockingSignals || 'n/a'}, review=${benchmarkReleaseRunbookReviewSignals || 'n/a'}, artifact=${benchmarkReleaseRunbookArtifact}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeApplicationStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeApplicationStatus || 'n/a'}, focus=${benchmarkReleaseRunbookKnowledgeApplicationFocusAreas || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeApplicationPriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeApplicationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeApplicationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatus || 'n/a'}, focus=${benchmarkReleaseRunbookKnowledgeRealdataCorrelationFocusAreas || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeRealdataCorrelationPriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeRealdataCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeRealdataCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainMatrixStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainMatrixStatus || 'n/a'}, focus=${benchmarkReleaseRunbookKnowledgeDomainMatrixFocusAreas || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeDomainMatrixPriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeDomainMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixStatus || 'n/a'}, focus=${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixFocusAreas || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixPriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftStatus = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_STATUS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftDomainRegressions = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_REGRESSIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftDomainImprovements = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_DOMAIN_IMPROVEMENTS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftRecommendations = envStr("STEP_BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_RECOMMENDATIONS", "");
    const benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftStatus || 'n/a'}, regressions=${benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftDomainRegressions || 'n/a'}, improvements=${benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftDomainImprovements || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainActionPlanStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseRunbookKnowledgeDomainActionPlanActions || 'n/a'}, priority_domains=${benchmarkReleaseRunbookKnowledgeDomainActionPlanPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanTotalActionCount || 'n/a'}, gate_open=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanGateOpen || 'n/a'}, priority_domains=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanRecommendedFirstActions || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneStatus || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDomains || 'n/a'}, release_blockers=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneReleaseBlockers || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftStatus || 'n/a'}, regressions=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftDomainRegressions || 'n/a'}, improvements=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftDomainImprovements || 'n/a'}, resolved_release_blockers=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftResolvedReleaseBlockers || 'n/a'}, new_release_blockers=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftNewReleaseBlockers || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatus || 'n/a'}, gate_open=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateGateOpen || 'n/a'}, releasable_domains=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateReleasableDomains || 'n/a'}, blocked_domains=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateBlockedDomains || 'n/a'}, priority_domains=${benchmarkReleaseRunbookKnowledgeDomainReleaseGatePriorityDomains || 'n/a'}, blocking_reasons=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateBlockingReasons || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeDomainReleaseGateRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus || 'n/a'}, summary=${benchmarkReleaseRunbookKnowledgeReferenceInventorySummary || 'n/a'}, priority_domains=${benchmarkReleaseRunbookKnowledgeReferenceInventoryPriorityDomains || 'n/a'}, refs=${benchmarkReleaseRunbookKnowledgeReferenceInventoryTotalReferenceItems || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeReferenceInventoryRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeSourceCoverageStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeSourceCoverageStatus || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeSourceCoverageDomainStatuses || 'n/a'}, expansion_candidates=${benchmarkReleaseRunbookKnowledgeSourceCoverageExpansionCandidates || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeSourceCoverageRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeSourceActionPlanStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeSourceActionPlanStatus || 'n/a'}, priority_domains=${benchmarkReleaseRunbookKnowledgeSourceActionPlanPriorityDomains || 'n/a'}, first_actions=${benchmarkReleaseRunbookKnowledgeSourceActionPlanRecommendedFirstActions || 'n/a'}, source_group_action_counts=${benchmarkReleaseRunbookKnowledgeSourceActionPlanSourceGroupActionCounts || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeSourceActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeSourceDriftStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeSourceDriftStatus || 'n/a'}, summary=${benchmarkReleaseRunbookKnowledgeSourceDriftSummary || 'n/a'}, source_group_regressions=${benchmarkReleaseRunbookKnowledgeSourceDriftSourceGroupRegressions || 'n/a'}, source_group_improvements=${benchmarkReleaseRunbookKnowledgeSourceDriftSourceGroupImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseRunbookKnowledgeSourceDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseRunbookKnowledgeSourceDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeSourceDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatus || 'n/a'}, focus=${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationFocusAreas || 'n/a'}, domains=${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationPriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationDomainStatuses || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookKnowledgeOutcomeDriftStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookKnowledgeOutcomeDriftStatus || 'n/a'}, summary=${benchmarkReleaseRunbookKnowledgeOutcomeDriftSummary || 'n/a'}, domain_regressions=${benchmarkReleaseRunbookKnowledgeOutcomeDriftDomainRegressions || 'n/a'}, domain_improvements=${benchmarkReleaseRunbookKnowledgeOutcomeDriftDomainImprovements || 'n/a'}, resolved_domains=${benchmarkReleaseRunbookKnowledgeOutcomeDriftResolvedPriorityDomains || 'n/a'}, new_domains=${benchmarkReleaseRunbookKnowledgeOutcomeDriftNewPriorityDomains || 'n/a'}, recommendations=${benchmarkReleaseRunbookKnowledgeOutcomeDriftRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookCompetitiveSurpassStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookCompetitiveSurpassStatus || 'n/a'}, gaps=${benchmarkReleaseRunbookCompetitiveSurpassPrimaryGaps || 'n/a'}, recommendations=${benchmarkReleaseRunbookCompetitiveSurpassRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookCompetitiveSurpassTrendStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookCompetitiveSurpassTrendStatus || 'n/a'}, summary=${benchmarkReleaseRunbookCompetitiveSurpassTrendSummary || 'n/a'}, recommendations=${benchmarkReleaseRunbookCompetitiveSurpassTrendRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatusLine = benchmarkReleaseRunbookEnabled
      ? `status=${benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatus || 'n/a'}, actions=${benchmarkReleaseRunbookCompetitiveSurpassActionPlanTotalActionCount || 'n/a'}, pillars=${benchmarkReleaseRunbookCompetitiveSurpassActionPlanPriorityPillars || 'n/a'}, recommendations=${benchmarkReleaseRunbookCompetitiveSurpassActionPlanRecommendations || 'n/a'}`
      : '⏭️ skipped';
    const benchmarkOperatorAdoptionStatusLine = benchmarkOperatorAdoptionEnabled
      ? `readiness=${benchmarkOperatorAdoptionReadiness}, mode=${benchmarkOperatorAdoptionMode}, next=${benchmarkOperatorAdoptionNextAction || 'n/a'}, automation_ready=${benchmarkOperatorAdoptionAutomationReady}, freeze_ready=${benchmarkOperatorAdoptionFreezeReady}, release=${benchmarkOperatorAdoptionReleaseStatus}, runbook=${benchmarkOperatorAdoptionRunbookStatus}, review_queue=${benchmarkOperatorAdoptionReviewQueueStatus}, feedback=${benchmarkOperatorAdoptionFeedbackStatus}, knowledge_drift=${benchmarkOperatorAdoptionKnowledgeDriftStatus}, knowledge_drift_summary=${benchmarkOperatorAdoptionKnowledgeDriftSummary || 'n/a'}, knowledge_outcome_drift=${benchmarkOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'}, knowledge_outcome_drift_summary=${benchmarkOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'}, blockers=${benchmarkOperatorAdoptionBlockingSignals || 'n/a'}, actions=${benchmarkOperatorAdoptionRecommendedActions || 'n/a'}, artifact=${benchmarkOperatorAdoptionArtifact}`
      : '⏭️ skipped';
    const assistantEvidenceStatus = assistantEvidenceEnabled
      ? `input=${assistantEvidenceInput}, records=${assistantEvidenceTotalRecords}, evidence_items=${assistantEvidenceTotalItems}, avg=${assistantEvidenceAverageCount}, evidence_cov=${assistantEvidenceCoverage}, decision_cov=${assistantDecisionPathCoverage}, source_cov=${assistantSourceSignalCoverage}`
      : '⏭️ skipped';
    const assistantEvidenceInsights = assistantEvidenceEnabled
      ? `kinds=${assistantTopRecordKinds || 'n/a'}, evidence=${assistantTopEvidenceTypes || 'n/a'}, sources=${assistantTopStructuredSources || 'n/a'}, missing=${assistantTopMissingFields || 'n/a'}`
      : '⏭️ skipped';
    const activeLearningReviewQueueStatusLine = activeLearningReviewQueueEnabled
      ? `input=${activeLearningReviewQueueInput}, total=${activeLearningReviewQueueTotal}, status=${activeLearningReviewQueueStatus}, critical=${activeLearningReviewQueueCriticalCount}, high=${activeLearningReviewQueueHighCount}, automation_ready=${activeLearningReviewQueueAutomationReadyCount}, critical_ratio=${activeLearningReviewQueueCriticalRatio}, high_ratio=${activeLearningReviewQueueHighRatio}, automation_ratio=${activeLearningReviewQueueAutomationReadyRatio}`
      : '⏭️ skipped';
    const activeLearningReviewQueueInsights = activeLearningReviewQueueEnabled
      ? `sample_types=${activeLearningReviewQueueTopSampleTypes || 'n/a'}, priorities=${activeLearningReviewQueueTopPriorities || 'n/a'}, sources=${activeLearningReviewQueueTopDecisionSources || 'n/a'}, reasons=${activeLearningReviewQueueTopReviewReasons || 'n/a'}`
      : '⏭️ skipped';
    const ocrReviewPackStatus = ocrReviewPackEnabled
      ? `input=${ocrReviewPackInput}, exported=${ocrExportedRecords}, review_candidates=${ocrReviewCandidateCount}, automation_ready=${ocrAutomationReadyCount}, readiness=${ocrAverageReadiness}, coverage=${ocrAverageCoverage}`
      : '⏭️ skipped';
    const ocrReviewPackInsights = ocrReviewPackEnabled
      ? `priorities=${ocrReviewPriorities || 'n/a'}, gaps=${ocrPrimaryGaps || 'n/a'}, reasons=${ocrReviewReasons || 'n/a'}, actions=${ocrRecommendedActions || 'n/a'}`
      : '⏭️ skipped';

    const reviewPackLight = !reviewPackEnabled
      ? '⚪'
      : (reviewCandidateCount > 0 ? '🟡' : '🟢');
    const reviewGateLight = !reviewGateEnabled
      ? '⚪'
      : (reviewGateStatusRaw === 'passed' ? '🟢' : '🔴');
    const trainSweepLight = !trainSweepEnabled
      ? '⚪'
      : (sweepTotalRunsInt <= 0 ? '🟡' : (sweepFailedRunsInt > 0 ? '🔴' : '🟢'));
    const benchmarkLight = !benchmarkScorecardEnabled
      ? '⚪'
      : (benchmarkOverallStatus.includes('gap') ? '🟡' : '🟢');
    const benchmarkScorecardOperatorAdoptionLight = !benchmarkScorecardEnabled
      ? '⚪'
      : (
        benchmarkScorecardOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkScorecardOperatorAdoptionStatus === 'missing' ||
            benchmarkScorecardOperatorAdoptionStatus === 'blocked' ||
            benchmarkScorecardOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkScorecardOperatorOutcomeDriftLight = !benchmarkScorecardEnabled
      ? '⚪'
      : (
        benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
            benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkEngineeringLight = !benchmarkEngineeringEnabled
      ? '⚪'
      : (
        benchmarkEngineeringArtifactStatus === 'engineering_semantics_ready'
          ? '🟢'
          : (
            benchmarkEngineeringArtifactStatus === 'weak_engineering_semantics'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkRealdataLight = !benchmarkRealdataEnabled
      ? '⚪'
      : (
        benchmarkRealdataStatus === 'realdata_foundation_ready'
          ? '🟢'
          : (
            benchmarkRealdataStatus === 'realdata_foundation_missing'
            || benchmarkRealdataStatus === 'environment_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkRealdataScorecardLight = !benchmarkRealdataScorecardEnabled
      ? '⚪'
      : (
        benchmarkRealdataScorecardStatus === 'realdata_scorecard_ready'
          ? '🟢'
          : (
            benchmarkRealdataScorecardStatus === 'realdata_scorecard_missing'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkCompetitiveSurpassLight = !benchmarkCompetitiveSurpassEnabled
      ? '⚪'
      : (
        benchmarkCompetitiveSurpassStatus === 'competitive_surpass_ready'
          ? '🟢'
          : (
            benchmarkCompetitiveSurpassStatus === 'competitive_surpass_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkCompetitiveSurpassTrendLight = !benchmarkCompetitiveSurpassTrendEnabled
      ? '⚪'
      : (
        benchmarkCompetitiveSurpassTrendStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkCompetitiveSurpassTrendStatus === 'improved' ||
            benchmarkCompetitiveSurpassTrendStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkCompetitiveSurpassActionPlanLight = !benchmarkCompetitiveSurpassActionPlanEnabled
      ? '⚪'
      : (
        benchmarkCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_ready'
          ? '🟢'
          : (
            benchmarkCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkKnowledgeLight = !benchmarkKnowledgeEnabled
      ? '⚪'
      : (
        benchmarkKnowledgeArtifactStatus === 'knowledge_foundation_ready'
          ? '🟢'
          : (
            benchmarkKnowledgeArtifactStatus === 'knowledge_foundation_missing'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkKnowledgeDriftLight = !benchmarkKnowledgeDriftEnabled
      ? '⚪'
      : (
        benchmarkKnowledgeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkKnowledgeDriftStatus === 'improved' ||
            benchmarkKnowledgeDriftStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkKnowledgeApplicationLight = !benchmarkKnowledgeApplicationEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeApplicationStatus === 'knowledge_application_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeApplicationStatus === 'knowledge_application_missing'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeRealdataCorrelationLight = !benchmarkKnowledgeRealdataCorrelationEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeRealdataCorrelationStatus === 'knowledge_realdata_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeRealdataCorrelationStatus === 'knowledge_realdata_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainMatrixLight = !benchmarkKnowledgeDomainMatrixEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainMatrixStatus === 'knowledge_domain_matrix_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainMatrixStatus === 'knowledge_domain_matrix_missing'
                || benchmarkKnowledgeDomainMatrixStatus === 'knowledge_domain_matrix_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainCapabilityMatrixLight = !benchmarkKnowledgeDomainCapabilityMatrixEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainCapabilityMatrixStatus === 'knowledge_domain_capability_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainCapabilityMatrixStatus === 'knowledge_domain_capability_missing'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainApiSurfaceMatrixLight = !benchmarkKnowledgeDomainApiSurfaceMatrixEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainApiSurfaceMatrixStatus === 'knowledge_domain_api_surface_matrix_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainApiSurfaceMatrixStatus === 'knowledge_domain_api_surface_matrix_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainCapabilityDriftLight = !benchmarkKnowledgeDomainCapabilityDriftEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainCapabilityDriftStatus === 'regressed'
            ? '🔴'
            : (
                benchmarkKnowledgeDomainCapabilityDriftStatus === 'improved'
                || benchmarkKnowledgeDomainCapabilityDriftStatus === 'stable'
                  ? '🟢'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainActionPlanLight = !benchmarkKnowledgeDomainActionPlanEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainActionPlanStatus === 'knowledge_domain_action_plan_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainActionPlanStatus === 'knowledge_domain_action_plan_blocked'
                  ? '🔴'
                  : '🟡'
                )
        );
    const benchmarkKnowledgeDomainSurfaceActionPlanLight = !benchmarkKnowledgeDomainSurfaceActionPlanEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainSurfaceActionPlanStatus === 'knowledge_domain_surface_action_plan_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainSurfaceActionPlanStatus === 'knowledge_domain_surface_action_plan_blocked'
                  ? '🔴'
                  : '🟡'
                )
        );
    const benchmarkKnowledgeDomainReleaseReadinessActionPlanLight = !benchmarkKnowledgeDomainReleaseReadinessActionPlanEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainReleaseReadinessActionPlanStatus === 'knowledge_domain_release_readiness_action_plan_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainReleaseReadinessActionPlanStatus === 'knowledge_domain_release_readiness_action_plan_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainControlPlaneLight = !benchmarkKnowledgeDomainControlPlaneEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainControlPlaneStatus === 'knowledge_domain_control_plane_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainControlPlaneStatus === 'knowledge_domain_control_plane_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainControlPlaneDriftLight = !benchmarkKnowledgeDomainControlPlaneDriftEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainControlPlaneDriftStatus === 'regressed'
            ? '🔴'
            : (
                benchmarkKnowledgeDomainControlPlaneDriftStatus === 'improved'
                || benchmarkKnowledgeDomainControlPlaneDriftStatus === 'stable'
                  ? '🟢'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainReleaseGateLight = !benchmarkKnowledgeDomainReleaseGateEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_blocked'
                || benchmarkKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_unavailable'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeDomainReleaseSurfaceAlignmentLight = !benchmarkKnowledgeDomainReleaseSurfaceAlignmentEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatus === 'aligned'
            ? '🟢'
            : (
                benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatus === 'diverged'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeReferenceInventoryLight = !benchmarkKnowledgeReferenceInventoryEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeSourceCoverageLight = !benchmarkKnowledgeSourceCoverageEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeSourceCoverageStatus === 'knowledge_source_coverage_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeSourceCoverageStatus === 'knowledge_source_coverage_missing'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeSourceActionPlanLight = !benchmarkKnowledgeSourceActionPlanEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeSourceActionPlanStatus === 'knowledge_source_action_plan_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeSourceActionPlanStatus === 'knowledge_source_action_plan_blocked'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeSourceDriftLight = !benchmarkKnowledgeSourceDriftEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeSourceDriftStatus === 'regressed'
            ? '🔴'
            : (
                benchmarkKnowledgeSourceDriftStatus === 'improved'
                || benchmarkKnowledgeSourceDriftStatus === 'stable'
                  ? '🟢'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeOutcomeCorrelationLight = !benchmarkKnowledgeOutcomeCorrelationEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeOutcomeCorrelationStatus === 'knowledge_outcome_correlation_ready'
            ? '🟢'
            : (
                benchmarkKnowledgeOutcomeCorrelationStatus === 'knowledge_outcome_correlation_missing'
                  ? '🔴'
                  : '🟡'
              )
        );
    const benchmarkKnowledgeOutcomeDriftLight = !benchmarkKnowledgeOutcomeDriftEnabled
      ? '⚪'
      : (
          benchmarkKnowledgeOutcomeDriftStatus === 'regressed'
            ? '🔴'
            : (
                benchmarkKnowledgeOutcomeDriftStatus === 'improved'
                || benchmarkKnowledgeOutcomeDriftStatus === 'stable'
                  ? '🟢'
                  : '🟡'
              )
        );
    const benchmarkOperationalLight = !benchmarkOperationalSummaryEnabled
      ? '⚪'
      : (benchmarkOperationalSummaryOverall === 'attention_required' ? '🟡' : '🟢');
    const benchmarkOperationalOperatorAdoptionLight = !benchmarkOperationalSummaryEnabled
      ? '⚪'
      : (
        benchmarkOperationalOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkOperationalOperatorAdoptionStatus === 'missing' ||
            benchmarkOperationalOperatorAdoptionStatus === 'blocked' ||
            benchmarkOperationalOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkOperationalOperatorOutcomeDriftLight = !benchmarkOperationalSummaryEnabled
      ? '⚪'
      : (
        benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
            benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkArtifactBundleLight = !benchmarkArtifactBundleEnabled
      ? '⚪'
      : (
        benchmarkArtifactBundleOverall === 'attention_required' ||
        benchmarkArtifactBundleOverall === 'gap_detected' ||
        !!benchmarkArtifactBundleBlockers
          ? '🟡'
          : '🟢'
      );
    const benchmarkArtifactBundleCompetitiveSurpassLight = !benchmarkArtifactBundleEnabled
      ? '⚪'
      : (
        benchmarkArtifactBundleCompetitiveSurpassStatus === 'competitive_surpass_ready'
          ? '🟢'
          : (
            benchmarkArtifactBundleCompetitiveSurpassStatus === 'competitive_surpass_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkArtifactBundleCompetitiveSurpassTrendLight = !benchmarkArtifactBundleEnabled
      ? '⚪'
      : (
        benchmarkArtifactBundleCompetitiveSurpassTrendStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkArtifactBundleCompetitiveSurpassTrendStatus === 'improved' ||
            benchmarkArtifactBundleCompetitiveSurpassTrendStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkArtifactBundleCompetitiveSurpassActionPlanLight = !benchmarkArtifactBundleEnabled
      ? '⚪'
      : (
        benchmarkArtifactBundleCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_ready'
          ? '🟢'
          : (
            benchmarkArtifactBundleCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkCompanionLight = !benchmarkCompanionSummaryEnabled
      ? '⚪'
      : (
        benchmarkCompanionSummaryOverall === 'attention_required' ||
        benchmarkCompanionReviewSurface === 'attention_required'
      ) ? '🟡' : '🟢';
    const benchmarkCompanionCompetitiveSurpassLight = !benchmarkCompanionSummaryEnabled
      ? '⚪'
      : (
        benchmarkCompanionCompetitiveSurpassStatus === 'competitive_surpass_ready'
          ? '🟢'
          : (
            benchmarkCompanionCompetitiveSurpassStatus === 'competitive_surpass_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkCompanionCompetitiveSurpassTrendLight = !benchmarkCompanionSummaryEnabled
      ? '⚪'
      : (
        benchmarkCompanionCompetitiveSurpassTrendStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkCompanionCompetitiveSurpassTrendStatus === 'improved' ||
            benchmarkCompanionCompetitiveSurpassTrendStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkCompanionCompetitiveSurpassActionPlanLight = !benchmarkCompanionSummaryEnabled
      ? '⚪'
      : (
        benchmarkCompanionCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_ready'
          ? '🟢'
          : (
            benchmarkCompanionCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseStatus === 'blocked'
          ? '🔴'
          : (benchmarkReleaseStatus === 'review_required' ? '🟡' : '🟢')
      );
    const benchmarkReleaseDecisionCompetitiveSurpassLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseCompetitiveSurpassStatus === 'competitive_surpass_ready'
          ? '🟢'
          : (
            benchmarkReleaseCompetitiveSurpassStatus === 'competitive_surpass_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionCompetitiveSurpassTrendLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseCompetitiveSurpassTrendStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseCompetitiveSurpassTrendStatus === 'improved' ||
            benchmarkReleaseCompetitiveSurpassTrendStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionCompetitiveSurpassActionPlanLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_ready'
          ? '🟢'
          : (
            benchmarkReleaseCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookFreezeReady === 'true'
          ? '🟢'
          : (benchmarkReleaseRunbookStatus === 'blocked' ? '🔴' : '🟡')
      );
    const benchmarkReleaseRunbookCompetitiveSurpassLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookCompetitiveSurpassStatus === 'competitive_surpass_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookCompetitiveSurpassStatus === 'competitive_surpass_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookCompetitiveSurpassTrendLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookCompetitiveSurpassTrendStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseRunbookCompetitiveSurpassTrendStatus === 'improved' ||
            benchmarkReleaseRunbookCompetitiveSurpassTrendStatus === 'stable'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookCompetitiveSurpassActionPlanLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatus === 'competitive_surpass_action_plan_blocked'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionScorecardOperatorAdoptionLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseScorecardOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkReleaseScorecardOperatorAdoptionStatus === 'missing' ||
            benchmarkReleaseScorecardOperatorAdoptionStatus === 'blocked' ||
            benchmarkReleaseScorecardOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionScorecardOperatorOutcomeDriftLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
            benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionOperationalOperatorAdoptionLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseOperationalOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkReleaseOperationalOperatorAdoptionStatus === 'missing' ||
            benchmarkReleaseOperationalOperatorAdoptionStatus === 'blocked' ||
            benchmarkReleaseOperationalOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionOperationalOperatorOutcomeDriftLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
                benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionReleaseSurfaceAlignmentLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus === 'aligned'
          ? '🟢'
          : (
            benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus === 'diverged'
              ? '🟡'
              : '🔴'
          )
      );
    const benchmarkReleaseDecisionKnowledgeDomainReleaseGateLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_ready'
          ? '🟢'
          : (
            benchmarkReleaseKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_blocked' ||
            benchmarkReleaseKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_unavailable'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseDecisionKnowledgeReferenceInventoryLight = !benchmarkReleaseDecisionEnabled
      ? '⚪'
      : (
        benchmarkReleaseKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_ready'
          ? '🟢'
          : (
            benchmarkReleaseKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_blocked' ||
            benchmarkReleaseKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_unavailable'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookScorecardOperatorAdoptionLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookScorecardOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookScorecardOperatorAdoptionStatus === 'missing' ||
            benchmarkReleaseRunbookScorecardOperatorAdoptionStatus === 'blocked' ||
            benchmarkReleaseRunbookScorecardOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookScorecardOperatorOutcomeDriftLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
            benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookOperationalOperatorAdoptionLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookOperationalOperatorAdoptionStatus === 'operator_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookOperationalOperatorAdoptionStatus === 'missing' ||
            benchmarkReleaseRunbookOperationalOperatorAdoptionStatus === 'blocked' ||
            benchmarkReleaseRunbookOperationalOperatorAdoptionStatus === 'unknown'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookOperationalOperatorOutcomeDriftLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'regressed'
          ? '🔴'
          : (
            benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'stable' ||
                benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus === 'improved'
              ? '🟢'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookReleaseSurfaceAlignmentLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus === 'aligned'
          ? '🟢'
          : (
            benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus === 'diverged'
              ? '🟡'
              : '🔴'
          )
      );
    const benchmarkReleaseRunbookKnowledgeDomainReleaseGateLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_blocked' ||
            benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatus === 'knowledge_domain_release_gate_unavailable'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkReleaseRunbookKnowledgeReferenceInventoryLight = !benchmarkReleaseRunbookEnabled
      ? '⚪'
      : (
        benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_ready'
          ? '🟢'
          : (
            benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_blocked' ||
            benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus === 'knowledge_reference_inventory_unavailable'
              ? '🔴'
              : '🟡'
          )
      );
    const benchmarkOperatorAdoptionLight = !benchmarkOperatorAdoptionEnabled
      ? '⚪'
      : (
        benchmarkOperatorAdoptionReadiness === 'operator_ready'
          ? '🟢'
          : (benchmarkOperatorAdoptionReadiness === 'blocked' ? '🔴' : '🟡')
      );
    const benchmarkOperatorAdoptionReleaseAlignmentLight = !benchmarkOperatorAdoptionEnabled
      ? '⚪'
      : (
        benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus === 'aligned'
          ? '🟢'
          : (
            benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus === 'diverged'
              ? '🟡'
              : '🔴'
          )
      );
    const assistantEvidenceLight = !assistantEvidenceEnabled
      ? '⚪'
      : (parseInt(assistantEvidenceTotalRecords || '0', 10) > 0 ? '🟢' : '🟡');
    const activeLearningReviewQueueLight = !activeLearningReviewQueueEnabled
      ? '⚪'
      : (activeLearningReviewQueueStatus === 'critical_backlog'
        ? '🔴'
        : (activeLearningReviewQueueStatus === 'managed_backlog' ? '🟡' : '🟢'));
    const ocrReviewPackLight = !ocrReviewPackEnabled
      ? '⚪'
      : (parseInt(ocrReviewCandidateCount || '0', 10) > 0 ? '🟡' : '🟢');
    const evalReportingStackSummaryPath = process.env.EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT || '';
    const evalReportingIndexPath = process.env.EVAL_REPORTING_INDEX_JSON_FOR_COMMENT || '';
    const loadJsonIfExists = (path) => {
      if (!path) {
        return null;
      }
      try {
        if (!fs.existsSync(path)) {
          return null;
        }
        return JSON.parse(fs.readFileSync(path, 'utf8'));
      } catch (_) {
        return null;
      }
    };
    const evalReportingStack = loadJsonIfExists(evalReportingStackSummaryPath);
    const evalReportingIndex = loadJsonIfExists(evalReportingIndexPath);
    let evalReportingSection = '';
    if (evalReportingStack && typeof evalReportingStack === 'object') {
      const stackStatus = String(evalReportingStack.status || 'unknown');
      const missingCount = Number(evalReportingStack.missing_count || 0);
      const staleCount = Number(evalReportingStack.stale_count || 0);
      const mismatchCount = Number(evalReportingStack.mismatch_count || 0);
      const landingPage = evalReportingIndex && typeof evalReportingIndex === 'object'
        ? String(evalReportingIndex.landing_page_html || '')
        : '';
      const staticReport = String(evalReportingStack.static_report_html || '');
      const interactiveReport = String(evalReportingStack.interactive_report_html || '');
      const stackLines = [
        '### Eval Reporting Stack',
        `- Status: ${stackStatus}`,
        `- Summary: missing=${missingCount}, stale=${staleCount}, mismatch=${mismatchCount}`,
      ];
      if (landingPage) {
        stackLines.push(`- Landing Page: ${landingPage}`);
      }
      if (staticReport) {
        stackLines.push(`- Static Report: ${staticReport}`);
      }
      if (interactiveReport) {
        stackLines.push(`- Interactive Report: ${interactiveReport}`);
      }
      evalReportingSection = `\n${stackLines.join('\n')}\n`;
    }

    const repositorySlug = process.env.GITHUB_REPOSITORY || `${context.repo.owner}/${context.repo.repo}`;
    const runId = process.env.GITHUB_RUN_ID || '';
    const runBaseUrl = runId
      ? `https://github.com/${repositorySlug}/actions/runs/${runId}`
      : '';

    const body = `## 📊 CAD ML Platform - Evaluation Results

    ${overallStatus}

    ### Scores
    | Module | Score | Threshold | Status |
    |--------|-------|-----------|--------|
    | **Combined** | ${combined.toFixed(3)} | ${minCombined} | ${combinedStatus} |
    | **Vision** | ${vision.toFixed(3)} | ${minVision} | ${visionStatus} |
    | **OCR** | ${ocr.toFixed(3)} | ${minOcr} | ${ocrStatus} |

    ### Formula
    \`Combined Score = 0.5 × Vision + 0.5 × OCR_normalized\`

    ### Additional Analysis
    | Check | Status |
    |-------|--------|
    | **Anomaly Detection** | ${hasAnomalies ? '⚠️ Anomalies detected' : '✅ No anomalies'} |
    | **Security Audit** | ${securityStatus === 'pass' ? '✅ Passed' : '⚠️ Issues found'} |
    | **Graph2D Review Pack** | ${reviewPackStatus} |
    | **Graph2D Review Insights** | ${reviewPackInsights} |
    | **Graph2D Review Gate** | ${reviewGateStatus} |
    | **Graph2D Review Gate Strict** | ${reviewGateStrictStatus} |
    | **Graph2D Train Sweep** | ${trainSweepStatus} |
    | **Benchmark Scorecard** | ${benchmarkScorecardStatus} |
    | **Benchmark Scorecard Operator Adoption** | ${benchmarkScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Scorecard Operator Outcome Drift** | ${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Engineering Signals** | ${benchmarkEngineeringStatusLine} |
    | **Benchmark Real-Data Signals** | ${benchmarkRealdataStatusLine} |
    | **Benchmark Real-Data Scorecard** | ${benchmarkRealdataScorecardStatusLine} |
    | **Benchmark Competitive Surpass** | ${benchmarkCompetitiveSurpassStatusLine} |
    | **Benchmark Competitive Surpass Trend** | ${benchmarkCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Competitive Surpass Action Plan** | ${benchmarkCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Knowledge Readiness** | ${benchmarkKnowledgeStatusLine} |
    | **Benchmark Knowledge Drift** | ${benchmarkKnowledgeDriftStatusLine} |
    | **Benchmark Knowledge Application** | ${benchmarkKnowledgeApplicationStatusLine} |
    | **Benchmark Knowledge Real-Data Correlation** | ${benchmarkKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Knowledge Domain Matrix** | ${benchmarkKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Knowledge Domain Capability Matrix** | ${benchmarkKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Knowledge Domain API Surface Matrix** | ${benchmarkKnowledgeDomainApiSurfaceMatrixStatusLine} |
    | **Benchmark Knowledge Domain Capability Drift** | ${benchmarkKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Knowledge Domain Action Plan** | ${benchmarkKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Surface Action Plan** | ${benchmarkKnowledgeDomainSurfaceActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Release Readiness Action Plan** | ${benchmarkKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Control Plane** | ${benchmarkKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Knowledge Domain Control Plane Drift** | ${benchmarkKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Knowledge Domain Release Gate** | ${benchmarkKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Knowledge Domain Release Surface Alignment** | ${benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatusLine} |
    | **Benchmark Knowledge Reference Inventory** | ${benchmarkKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Knowledge Source Coverage** | ${benchmarkKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Knowledge Source Action Plan** | ${benchmarkKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Knowledge Source Drift** | ${benchmarkKnowledgeSourceDriftStatusLine} |
    | **Benchmark Knowledge Outcome Correlation** | ${benchmarkKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Knowledge Outcome Drift** | ${benchmarkKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Knowledge Focus Areas** | ${benchmarkKnowledgeFocusAreas || benchmarkKnowledgeFocusAreasScorecard || 'n/a'} |
    | **Benchmark Knowledge Domains** | ${benchmarkKnowledgePriorityDomains || 'n/a'} / ${benchmarkKnowledgeDomainFocusAreas || 'n/a'} |
    | **Benchmark Knowledge Recommendations** | ${benchmarkKnowledgeRecommendations || benchmarkKnowledgeReferenceItems || 'n/a'} |
    | **Benchmark Knowledge Application Recommendations** | ${benchmarkKnowledgeApplicationRecommendations || 'n/a'} |
    | **Benchmark Knowledge Real-Data Recommendations** | ${benchmarkKnowledgeRealdataCorrelationRecommendations || 'n/a'} |
    | **Benchmark Knowledge Outcome Recommendations** | ${benchmarkKnowledgeOutcomeCorrelationRecommendations || 'n/a'} |
    | **Benchmark Knowledge Drift Recommendations** | ${benchmarkKnowledgeDriftRecommendations || 'n/a'} |
    | **Benchmark Engineering Recommendations** | ${benchmarkEngineeringRecommendations || benchmarkEngineeringTopStandardTypes || 'n/a'} |
    | **Benchmark Recommendations** | ${benchmarkRecommendations || 'n/a'} |
    | **Benchmark Feedback Flywheel** | ${benchmarkFeedbackFlywheelStatus || 'n/a'} |
    | **Feedback Flywheel Artifact** | ${feedbackFlywheelBenchmarkStatusLine} |
    | **Benchmark Operational Summary** | ${benchmarkOperationalSummaryStatus} |
    | **Benchmark Operational Operator Adoption** | ${benchmarkOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Operational Operator Outcome Drift** | ${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Knowledge Drift** | ${benchmarkArtifactBundleKnowledgeDriftStatus || 'n/a'} / ${benchmarkArtifactBundleKnowledgeDriftRecommendations || 'n/a'} |
    | **Benchmark Artifact Bundle** | ${benchmarkArtifactBundleStatus} |
    | **Benchmark Artifact Bundle Knowledge Application** | ${benchmarkArtifactBundleKnowledgeApplicationStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Real-Data** | ${benchmarkArtifactBundleKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Matrix** | ${benchmarkArtifactBundleKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Capability Matrix** | ${benchmarkArtifactBundleKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Capability Drift** | ${benchmarkArtifactBundleKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Action Plan** | ${benchmarkArtifactBundleKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Release Readiness Action Plan** | ${benchmarkArtifactBundleKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Control Plane** | ${benchmarkArtifactBundleKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Control Plane Drift** | ${benchmarkArtifactBundleKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domain Release Gate** | ${benchmarkArtifactBundleKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Source Coverage** | ${benchmarkArtifactBundleKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Source Action Plan** | ${benchmarkArtifactBundleKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Source Drift** | ${benchmarkArtifactBundleKnowledgeSourceDriftStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Outcome Correlation** | ${benchmarkArtifactBundleKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Outcome Drift** | ${benchmarkArtifactBundleKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Artifact Bundle Knowledge Domains** | ${benchmarkArtifactBundleKnowledgePriorityDomains || 'n/a'} / ${benchmarkArtifactBundleKnowledgeDomainFocusAreas || 'n/a'} |
    | **Benchmark Artifact Bundle Real-Data** | ${benchmarkArtifactBundleRealdataStatus || 'n/a'} / ${benchmarkArtifactBundleRealdataRecommendations || 'n/a'} |
    | **Benchmark Artifact Bundle Real-Data Scorecard** | ${benchmarkArtifactBundleRealdataScorecardStatus || 'n/a'} / ${benchmarkArtifactBundleRealdataScorecardRecommendations || 'n/a'} |
    | **Benchmark Artifact Bundle Competitive Surpass** | ${benchmarkArtifactBundleCompetitiveSurpassStatusLine} |
    | **Benchmark Artifact Bundle Competitive Surpass Trend** | ${benchmarkArtifactBundleCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Artifact Bundle Competitive Surpass Action Plan** | ${benchmarkArtifactBundleCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Companion Knowledge Drift** | ${benchmarkCompanionKnowledgeDriftStatus || 'n/a'} / ${benchmarkCompanionKnowledgeDriftRecommendations || 'n/a'} |
    | **Benchmark Companion Summary** | ${benchmarkCompanionSummaryStatus} |
    | **Benchmark Companion Knowledge Application** | ${benchmarkCompanionKnowledgeApplicationStatusLine} |
    | **Benchmark Companion Knowledge Real-Data** | ${benchmarkCompanionKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Companion Knowledge Domain Matrix** | ${benchmarkCompanionKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Companion Knowledge Domain Capability Matrix** | ${benchmarkCompanionKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Companion Knowledge Domain Capability Drift** | ${benchmarkCompanionKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Companion Knowledge Domain Action Plan** | ${benchmarkCompanionKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Companion Knowledge Domain Release Readiness Action Plan** | ${benchmarkCompanionKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Companion Knowledge Domain Control Plane** | ${benchmarkCompanionKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Companion Knowledge Domain Control Plane Drift** | ${benchmarkCompanionKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Companion Knowledge Domain Release Gate** | ${benchmarkCompanionKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Companion Knowledge Source Coverage** | ${benchmarkCompanionKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Companion Knowledge Source Action Plan** | ${benchmarkCompanionKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Companion Knowledge Source Drift** | ${benchmarkCompanionKnowledgeSourceDriftStatusLine} |
    | **Benchmark Companion Knowledge Outcome Correlation** | ${benchmarkCompanionKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Companion Knowledge Outcome Drift** | ${benchmarkCompanionKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Companion Knowledge Domains** | ${benchmarkCompanionKnowledgePriorityDomains || 'n/a'} / ${benchmarkCompanionKnowledgeDomainFocusAreas || 'n/a'} |
    | **Benchmark Companion Real-Data** | ${benchmarkCompanionRealdataStatus || 'n/a'} / ${benchmarkCompanionRealdataRecommendations || 'n/a'} |
    | **Benchmark Companion Real-Data Scorecard** | ${benchmarkCompanionRealdataScorecardStatus || 'n/a'} / ${benchmarkCompanionRealdataScorecardRecommendations || 'n/a'} |
    | **Benchmark Companion Competitive Surpass** | ${benchmarkCompanionCompetitiveSurpassStatusLine} |
    | **Benchmark Companion Competitive Surpass Trend** | ${benchmarkCompanionCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Companion Competitive Surpass Action Plan** | ${benchmarkCompanionCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Decision Knowledge Drift** | ${benchmarkReleaseKnowledgeDriftStatus || 'n/a'} / ${benchmarkReleaseKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision** | ${benchmarkReleaseDecisionStatus} |
    | **Benchmark Release Decision Competitive Surpass** | ${benchmarkReleaseCompetitiveSurpassStatusLine} |
    | **Benchmark Release Decision Competitive Surpass Trend** | ${benchmarkReleaseCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Release Decision Competitive Surpass Action Plan** | ${benchmarkReleaseCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Decision Knowledge Application** | ${benchmarkReleaseKnowledgeApplicationStatusLine} |
    | **Benchmark Release Decision Knowledge Real-Data** | ${benchmarkReleaseKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Matrix** | ${benchmarkReleaseKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Capability Matrix** | ${benchmarkReleaseKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Capability Drift** | ${benchmarkReleaseKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Action Plan** | ${benchmarkReleaseKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Release Readiness Action Plan** | ${benchmarkReleaseKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Control Plane** | ${benchmarkReleaseKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Control Plane Drift** | ${benchmarkReleaseKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Release Decision Knowledge Domain Release Gate** | ${benchmarkReleaseKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Release Decision Knowledge Source Coverage** | ${benchmarkReleaseKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Release Decision Knowledge Source Action Plan** | ${benchmarkReleaseKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Release Decision Knowledge Source Drift** | ${benchmarkReleaseKnowledgeSourceDriftStatusLine} |
    | **Benchmark Release Decision Knowledge Outcome Correlation** | ${benchmarkReleaseKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Release Decision Knowledge Outcome Drift** | ${benchmarkReleaseKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Release Decision Knowledge Domains** | ${benchmarkReleaseKnowledgePriorityDomains || 'n/a'} / ${benchmarkReleaseKnowledgeDomainFocusAreas || 'n/a'} |
    | **Benchmark Release Decision Real-Data** | ${benchmarkReleaseRealdataStatus || 'n/a'} / ${benchmarkReleaseRealdataRecommendations || 'n/a'} |
    | **Benchmark Release Decision Real-Data Scorecard** | ${benchmarkReleaseRealdataScorecardStatus || 'n/a'} / ${benchmarkReleaseRealdataScorecardRecommendations || 'n/a'} |
    | **Benchmark Release Decision Scorecard Operator Adoption** | ${benchmarkReleaseScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkReleaseScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Release Decision Scorecard Operator Outcome Drift** | ${benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision Operational Operator Adoption** | ${benchmarkReleaseOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Release Decision Operational Operator Outcome Drift** | ${benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision Release Surface Alignment** | ${benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Release Decision Release Surface Mismatches** | ${benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Release Decision Knowledge Reference Inventory** | ${benchmarkReleaseKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Release Runbook Knowledge Drift** | ${benchmarkReleaseRunbookKnowledgeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Operator Drift** | ${benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftStatus || 'n/a'} / ${benchmarkArtifactBundleOperatorAdoptionKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Operator Outcome Drift** | ${benchmarkArtifactBundleOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkArtifactBundleOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Scorecard Operator Adoption** | ${benchmarkArtifactBundleScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkArtifactBundleScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Artifact Bundle Scorecard Operator Outcome Drift** | ${benchmarkArtifactBundleScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkArtifactBundleScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Operational Operator Adoption** | ${benchmarkArtifactBundleOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Artifact Bundle Operational Operator Outcome Drift** | ${benchmarkArtifactBundleOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkArtifactBundleOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Release Surface Alignment** | ${benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Release Surface Mismatches** | ${benchmarkArtifactBundleOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Artifact Bundle Knowledge Domain Release Surface Alignment** | ${benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Artifact Bundle Knowledge Domain Release Surface Mismatches** | ${benchmarkArtifactBundleKnowledgeDomainReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Artifact Bundle Knowledge Reference Inventory** | ${benchmarkArtifactBundleKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Companion Operator Drift** | ${benchmarkCompanionOperatorAdoptionKnowledgeDriftStatus || 'n/a'} / ${benchmarkCompanionOperatorAdoptionKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Companion Operator Outcome Drift** | ${benchmarkCompanionOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkCompanionOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Companion Scorecard Operator Adoption** | ${benchmarkCompanionScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkCompanionScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Companion Scorecard Operator Outcome Drift** | ${benchmarkCompanionScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkCompanionScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Companion Operational Operator Adoption** | ${benchmarkCompanionOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Companion Operational Operator Outcome Drift** | ${benchmarkCompanionOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkCompanionOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Companion Release Surface Alignment** | ${benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Companion Release Surface Mismatches** | ${benchmarkCompanionOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Companion Knowledge Domain Release Surface Alignment** | ${benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Companion Knowledge Domain Release Surface Mismatches** | ${benchmarkCompanionKnowledgeDomainReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Companion Knowledge Reference Inventory** | ${benchmarkCompanionKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Release Decision Operator Drift** | ${benchmarkReleaseOperatorAdoptionKnowledgeDriftStatus || 'n/a'} / ${benchmarkReleaseOperatorAdoptionKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision Operator Outcome Drift** | ${benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook** | ${benchmarkReleaseRunbookStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass** | ${benchmarkReleaseRunbookCompetitiveSurpassStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass Trend** | ${benchmarkReleaseRunbookCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass Action Plan** | ${benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Runbook Knowledge Application** | ${benchmarkReleaseRunbookKnowledgeApplicationStatusLine} |
    | **Benchmark Release Runbook Knowledge Real-Data** | ${benchmarkReleaseRunbookKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Matrix** | ${benchmarkReleaseRunbookKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Capability Matrix** | ${benchmarkReleaseRunbookKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Capability Drift** | ${benchmarkReleaseRunbookKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Action Plan** | ${benchmarkReleaseRunbookKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Release Readiness Action Plan** | ${benchmarkReleaseRunbookKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Control Plane** | ${benchmarkReleaseRunbookKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Control Plane Drift** | ${benchmarkReleaseRunbookKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Release Runbook Knowledge Domain Release Gate** | ${benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Release Runbook Knowledge Source Coverage** | ${benchmarkReleaseRunbookKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Release Runbook Knowledge Source Action Plan** | ${benchmarkReleaseRunbookKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Release Runbook Knowledge Source Drift** | ${benchmarkReleaseRunbookKnowledgeSourceDriftStatusLine} |
    | **Benchmark Release Runbook Knowledge Outcome Correlation** | ${benchmarkReleaseRunbookKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Release Runbook Knowledge Outcome Drift** | ${benchmarkReleaseRunbookKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Release Runbook Knowledge Domains** | ${benchmarkReleaseRunbookKnowledgePriorityDomains || 'n/a'} / ${benchmarkReleaseRunbookKnowledgeDomainFocusAreas || 'n/a'} |
    | **Benchmark Release Runbook Real-Data** | ${benchmarkReleaseRunbookRealdataStatus || 'n/a'} / ${benchmarkReleaseRunbookRealdataRecommendations || 'n/a'} |
    | **Benchmark Release Runbook Real-Data Scorecard** | ${benchmarkReleaseRunbookRealdataScorecardStatus || 'n/a'} / ${benchmarkReleaseRunbookRealdataScorecardRecommendations || 'n/a'} |
    | **Benchmark Release Runbook Scorecard Operator Adoption** | ${benchmarkReleaseRunbookScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkReleaseRunbookScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Release Runbook Scorecard Operator Outcome Drift** | ${benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Operational Operator Adoption** | ${benchmarkReleaseRunbookOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Release Runbook Operational Operator Outcome Drift** | ${benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Release Surface Alignment** | ${benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Release Runbook Release Surface Mismatches** | ${benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Benchmark Release Runbook Operator Drift** | ${benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookOperatorAdoptionKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Operator Outcome Drift** | ${benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Knowledge Reference Inventory** | ${benchmarkReleaseRunbookKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Operator Adoption** | ${benchmarkOperatorAdoptionStatusLine} |
    | **Benchmark Operator Adoption Knowledge Drift** | ${benchmarkOperatorAdoptionKnowledgeDriftStatus || 'n/a'} / ${benchmarkOperatorAdoptionKnowledgeDriftSummary || 'n/a'} |
    | **Benchmark Operator Adoption Knowledge Outcome Drift** | ${benchmarkOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Operator Adoption Release Surface Alignment** | ${benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Operator Adoption Release Surface Mismatches** | ${benchmarkOperatorAdoptionReleaseSurfaceAlignmentMismatches || 'n/a'} |
    | **Assistant Evidence Report** | ${assistantEvidenceStatus} |
    | **Assistant Evidence Insights** | ${assistantEvidenceInsights} |
    | **Active-Learning Review Queue** | ${activeLearningReviewQueueStatusLine} |
    | **Active-Learning Review Queue Insights** | ${activeLearningReviewQueueInsights} |
    | **OCR Review Pack** | ${ocrReviewPackStatus} |
    | **OCR Review Insights** | ${ocrReviewPackInsights} |

    ### Graph2D Signal Lights
    | Signal | State | Detail |
    |--------|-------|--------|
    | **Review Pack** | ${reviewPackLight} | ${reviewPackStatus} |
    | **Review Gate** | ${reviewGateLight} | ${reviewGateStatus} |
    | **Train Sweep** | ${trainSweepLight} | ${trainSweepStatus} |
    | **Benchmark Scorecard** | ${benchmarkLight} | ${benchmarkScorecardStatus} |
    | **Benchmark Scorecard Operator Adoption** | ${benchmarkScorecardOperatorAdoptionLight} | ${benchmarkScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Scorecard Operator Outcome Drift** | ${benchmarkScorecardOperatorOutcomeDriftLight} | ${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Engineering Signals** | ${benchmarkEngineeringLight} | ${benchmarkEngineeringStatusLine} |
    | **Benchmark Real-Data Signals** | ${benchmarkRealdataLight} | ${benchmarkRealdataStatusLine} |
    | **Benchmark Real-Data Scorecard** | ${benchmarkRealdataScorecardLight} | ${benchmarkRealdataScorecardStatusLine} |
    | **Benchmark Competitive Surpass** | ${benchmarkCompetitiveSurpassLight} | ${benchmarkCompetitiveSurpassStatusLine} |
    | **Benchmark Competitive Surpass Trend** | ${benchmarkCompetitiveSurpassTrendLight} | ${benchmarkCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Competitive Surpass Action Plan** | ${benchmarkCompetitiveSurpassActionPlanLight} | ${benchmarkCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Knowledge Readiness** | ${benchmarkKnowledgeLight} | ${benchmarkKnowledgeStatusLine} |
    | **Benchmark Knowledge Drift** | ${benchmarkKnowledgeDriftLight} | ${benchmarkKnowledgeDriftStatusLine} |
    | **Benchmark Knowledge Application** | ${benchmarkKnowledgeApplicationLight} | ${benchmarkKnowledgeApplicationStatusLine} |
    | **Benchmark Knowledge Real-Data Correlation** | ${benchmarkKnowledgeRealdataCorrelationLight} | ${benchmarkKnowledgeRealdataCorrelationStatusLine} |
    | **Benchmark Knowledge Domain Matrix** | ${benchmarkKnowledgeDomainMatrixLight} | ${benchmarkKnowledgeDomainMatrixStatusLine} |
    | **Benchmark Knowledge Domain Capability Matrix** | ${benchmarkKnowledgeDomainCapabilityMatrixLight} | ${benchmarkKnowledgeDomainCapabilityMatrixStatusLine} |
    | **Benchmark Knowledge Domain API Surface Matrix** | ${benchmarkKnowledgeDomainApiSurfaceMatrixLight} | ${benchmarkKnowledgeDomainApiSurfaceMatrixStatusLine} |
    | **Benchmark Knowledge Domain Capability Drift** | ${benchmarkKnowledgeDomainCapabilityDriftLight} | ${benchmarkKnowledgeDomainCapabilityDriftStatusLine} |
    | **Benchmark Knowledge Domain Action Plan** | ${benchmarkKnowledgeDomainActionPlanLight} | ${benchmarkKnowledgeDomainActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Surface Action Plan** | ${benchmarkKnowledgeDomainSurfaceActionPlanLight} | ${benchmarkKnowledgeDomainSurfaceActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Release Readiness Action Plan** | ${benchmarkKnowledgeDomainReleaseReadinessActionPlanLight} | ${benchmarkKnowledgeDomainReleaseReadinessActionPlanStatusLine} |
    | **Benchmark Knowledge Domain Control Plane** | ${benchmarkKnowledgeDomainControlPlaneLight} | ${benchmarkKnowledgeDomainControlPlaneStatusLine} |
    | **Benchmark Knowledge Domain Control Plane Drift** | ${benchmarkKnowledgeDomainControlPlaneDriftLight} | ${benchmarkKnowledgeDomainControlPlaneDriftStatusLine} |
    | **Benchmark Knowledge Domain Release Gate** | ${benchmarkKnowledgeDomainReleaseGateLight} | ${benchmarkKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Knowledge Domain Release Surface Alignment** | ${benchmarkKnowledgeDomainReleaseSurfaceAlignmentLight} | ${benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatusLine} |
    | **Benchmark Knowledge Domain Release Gate** | ${benchmarkKnowledgeDomainReleaseGateLight} | ${benchmarkKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Knowledge Domain Release Surface Alignment** | ${benchmarkKnowledgeDomainReleaseSurfaceAlignmentLight} | ${benchmarkKnowledgeDomainReleaseSurfaceAlignmentStatusLine} |
    | **Benchmark Knowledge Reference Inventory** | ${benchmarkKnowledgeReferenceInventoryLight} | ${benchmarkKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Knowledge Source Coverage** | ${benchmarkKnowledgeSourceCoverageLight} | ${benchmarkKnowledgeSourceCoverageStatusLine} |
    | **Benchmark Knowledge Source Action Plan** | ${benchmarkKnowledgeSourceActionPlanLight} | ${benchmarkKnowledgeSourceActionPlanStatusLine} |
    | **Benchmark Knowledge Source Drift** | ${benchmarkKnowledgeSourceDriftLight} | ${benchmarkKnowledgeSourceDriftStatusLine} |
    | **Benchmark Knowledge Outcome Correlation** | ${benchmarkKnowledgeOutcomeCorrelationLight} | ${benchmarkKnowledgeOutcomeCorrelationStatusLine} |
    | **Benchmark Knowledge Outcome Drift** | ${benchmarkKnowledgeOutcomeDriftLight} | ${benchmarkKnowledgeOutcomeDriftStatusLine} |
    | **Benchmark Operational Summary** | ${benchmarkOperationalLight} | ${benchmarkOperationalSummaryStatus} |
    | **Benchmark Operational Operator Adoption** | ${benchmarkOperationalOperatorAdoptionLight} | ${benchmarkOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Operational Operator Outcome Drift** | ${benchmarkOperationalOperatorOutcomeDriftLight} | ${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Artifact Bundle** | ${benchmarkArtifactBundleLight} | ${benchmarkArtifactBundleStatus} |
    | **Benchmark Artifact Bundle Competitive Surpass** | ${benchmarkArtifactBundleCompetitiveSurpassLight} | ${benchmarkArtifactBundleCompetitiveSurpassStatusLine} |
    | **Benchmark Artifact Bundle Competitive Surpass Trend** | ${benchmarkArtifactBundleCompetitiveSurpassTrendLight} | ${benchmarkArtifactBundleCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Artifact Bundle Competitive Surpass Action Plan** | ${benchmarkArtifactBundleCompetitiveSurpassActionPlanLight} | ${benchmarkArtifactBundleCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Companion Summary** | ${benchmarkCompanionLight} | ${benchmarkCompanionSummaryStatus} |
    | **Benchmark Companion Competitive Surpass** | ${benchmarkCompanionCompetitiveSurpassLight} | ${benchmarkCompanionCompetitiveSurpassStatusLine} |
    | **Benchmark Companion Competitive Surpass Trend** | ${benchmarkCompanionCompetitiveSurpassTrendLight} | ${benchmarkCompanionCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Companion Competitive Surpass Action Plan** | ${benchmarkCompanionCompetitiveSurpassActionPlanLight} | ${benchmarkCompanionCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Decision** | ${benchmarkReleaseDecisionLight} | ${benchmarkReleaseDecisionStatus} |
    | **Benchmark Release Decision Competitive Surpass** | ${benchmarkReleaseDecisionCompetitiveSurpassLight} | ${benchmarkReleaseCompetitiveSurpassStatusLine} |
    | **Benchmark Release Decision Competitive Surpass Trend** | ${benchmarkReleaseDecisionCompetitiveSurpassTrendLight} | ${benchmarkReleaseCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Release Decision Competitive Surpass Action Plan** | ${benchmarkReleaseDecisionCompetitiveSurpassActionPlanLight} | ${benchmarkReleaseCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Decision Scorecard Operator Adoption** | ${benchmarkReleaseDecisionScorecardOperatorAdoptionLight} | ${benchmarkReleaseScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkReleaseScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Release Decision Scorecard Operator Outcome Drift** | ${benchmarkReleaseDecisionScorecardOperatorOutcomeDriftLight} | ${benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision Operational Operator Adoption** | ${benchmarkReleaseDecisionOperationalOperatorAdoptionLight} | ${benchmarkReleaseOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Release Decision Operational Operator Outcome Drift** | ${benchmarkReleaseDecisionOperationalOperatorOutcomeDriftLight} | ${benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Decision Release Surface Alignment** | ${benchmarkReleaseDecisionReleaseSurfaceAlignmentLight} | ${benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkReleaseOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Release Decision Knowledge Domain Release Gate** | ${benchmarkReleaseDecisionKnowledgeDomainReleaseGateLight} | ${benchmarkReleaseKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Release Decision Knowledge Reference Inventory** | ${benchmarkReleaseDecisionKnowledgeReferenceInventoryLight} | ${benchmarkReleaseKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Release Runbook** | ${benchmarkReleaseRunbookLight} | ${benchmarkReleaseRunbookStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass** | ${benchmarkReleaseRunbookCompetitiveSurpassLight} | ${benchmarkReleaseRunbookCompetitiveSurpassStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass Trend** | ${benchmarkReleaseRunbookCompetitiveSurpassTrendLight} | ${benchmarkReleaseRunbookCompetitiveSurpassTrendStatusLine} |
    | **Benchmark Release Runbook Competitive Surpass Action Plan** | ${benchmarkReleaseRunbookCompetitiveSurpassActionPlanLight} | ${benchmarkReleaseRunbookCompetitiveSurpassActionPlanStatusLine} |
    | **Benchmark Release Runbook Scorecard Operator Adoption** | ${benchmarkReleaseRunbookScorecardOperatorAdoptionLight} | ${benchmarkReleaseRunbookScorecardOperatorAdoptionStatus || 'n/a'} / ${benchmarkReleaseRunbookScorecardOperatorAdoptionMode || 'n/a'} |
    | **Benchmark Release Runbook Scorecard Operator Outcome Drift** | ${benchmarkReleaseRunbookScorecardOperatorOutcomeDriftLight} | ${benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookScorecardOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Operational Operator Adoption** | ${benchmarkReleaseRunbookOperationalOperatorAdoptionLight} | ${benchmarkReleaseRunbookOperationalOperatorAdoptionStatus || 'n/a'} |
    | **Benchmark Release Runbook Operational Operator Outcome Drift** | ${benchmarkReleaseRunbookOperationalOperatorOutcomeDriftLight} | ${benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftStatus || 'n/a'} / ${benchmarkReleaseRunbookOperationalOperatorAdoptionKnowledgeOutcomeDriftSummary || 'n/a'} |
    | **Benchmark Release Runbook Release Surface Alignment** | ${benchmarkReleaseRunbookReleaseSurfaceAlignmentLight} | ${benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkReleaseRunbookOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Benchmark Release Runbook Knowledge Domain Release Gate** | ${benchmarkReleaseRunbookKnowledgeDomainReleaseGateLight} | ${benchmarkReleaseRunbookKnowledgeDomainReleaseGateStatusLine} |
    | **Benchmark Release Runbook Knowledge Reference Inventory** | ${benchmarkReleaseRunbookKnowledgeReferenceInventoryLight} | ${benchmarkReleaseRunbookKnowledgeReferenceInventoryStatusLine} |
    | **Benchmark Operator Adoption** | ${benchmarkOperatorAdoptionLight} | ${benchmarkOperatorAdoptionStatusLine} |
    | **Benchmark Operator Adoption Release Surface Alignment** | ${benchmarkOperatorAdoptionReleaseAlignmentLight} | ${benchmarkOperatorAdoptionReleaseSurfaceAlignmentStatus || 'n/a'} / ${benchmarkOperatorAdoptionReleaseSurfaceAlignmentSummary || 'n/a'} |
    | **Assistant Evidence Report** | ${assistantEvidenceLight} | ${assistantEvidenceStatus} |
    | **Active-Learning Review Queue** | ${activeLearningReviewQueueLight} | ${activeLearningReviewQueueStatusLine} |
    | **OCR Review Pack** | ${ocrReviewPackLight} | ${ocrReviewPackStatus} |

    ${evalReportingSection}

    ### Quick Actions
    - 📋 [View Full Report](${runBaseUrl || '#'})
    - 📈 [Download Artifacts](${runBaseUrl ? `${runBaseUrl}#artifacts` : '#'})
    - 🔍 [Check Logs](${runBaseUrl ? `${runBaseUrl}/jobs` : '#'})

    ---
    *Updated: ${new Date().toISOString().replace('T', ' ').substring(0, 19)} UTC*
    *Commit: ${context.sha.substring(0, 7)}*`;

    const { data: comments } = await github.rest.issues.listComments({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number
    });

    const botComment = comments.find(comment =>
      comment.user.type === 'Bot' &&
      comment.body.includes('CAD ML Platform - Evaluation Results')
    );

    if (botComment) {
      await github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: botComment.id,
        body: body
      });
    } else {
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body: body
      });
    }
  } catch (error) {
    console.warn(`PR comment skipped: ${error.message}`);
  }
}

module.exports = { commentEvaluationReportPR };
