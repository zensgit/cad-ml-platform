#!/usr/bin/env bash
set -euo pipefail

eval "$(python3 - <<'INNER'
import json
import os
import shlex

def step_output(steps, step_id, output_name):
    step = steps.get(step_id) or {}
    outputs = step.get('outputs') or {}
    value = outputs.get(output_name)
    if value is None:
        return ''
    return str(value)

inputs = json.loads(os.environ.get('GITHUB_EVENT_INPUTS_JSON') or '{}')
steps = json.loads(os.environ.get('GITHUB_STEPS_JSON') or '{}')
mapping = {"WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ENABLE": ["inputs", "benchmark_artifact_bundle_enable"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_JSON": ["inputs", "benchmark_artifact_bundle_scorecard_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_SUMMARY_JSON": ["inputs", "benchmark_artifact_bundle_operational_summary_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_JSON": ["inputs", "benchmark_artifact_bundle_feedback_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_JSON": ["inputs", "benchmark_artifact_bundle_assistant_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_JSON": ["inputs", "benchmark_artifact_bundle_review_queue_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_OCR_REVIEW_JSON": ["inputs", "benchmark_artifact_bundle_ocr_review_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPANION_SUMMARY_JSON": ["inputs", "benchmark_artifact_bundle_companion_summary_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_RELEASE_DECISION_JSON": ["inputs", "benchmark_artifact_bundle_release_decision_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_SIGNALS_JSON": ["inputs", "benchmark_artifact_bundle_engineering_signals_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SIGNALS_JSON": ["inputs", "benchmark_artifact_bundle_realdata_signals_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_JSON": ["inputs", "benchmark_artifact_bundle_realdata_scorecard_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_READINESS_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_readiness_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_drift_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_application_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_realdata_correlation_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_matrix_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_capability_matrix_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_capability_drift_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_control_plane_drift_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_release_surface_alignment_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_action_plan_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_source_coverage_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_source_action_plan_json"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_OUTPUT_JSON": ["steps", "benchmark_knowledge_source_drift", "output_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_outcome_correlation_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_outcome_drift_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_JSON": ["inputs", "benchmark_artifact_bundle_competitive_surpass_index_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_JSON": ["inputs", "benchmark_artifact_bundle_competitive_surpass_trend_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_JSON": ["inputs", "benchmark_artifact_bundle_competitive_surpass_action_plan_json"], "STEP_BENCHMARK_SCORECARD_OUTPUT_JSON": ["steps", "benchmark_scorecard", "output_json"], "STEP_BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_JSON": ["steps", "benchmark_operational_summary", "output_json"], "STEP_FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_JSON": ["steps", "feedback_flywheel_benchmark", "output_json"], "STEP_ASSISTANT_EVIDENCE_REPORT_OUTPUT_JSON": ["steps", "assistant_evidence_report", "output_json"], "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON": ["steps", "active_learning_review_queue_report", "output_json"], "STEP_OCR_REVIEW_PACK_OUTPUT_JSON": ["steps", "ocr_review_pack", "output_json"], "STEP_BENCHMARK_COMPANION_SUMMARY_OUTPUT_JSON": ["steps", "benchmark_companion_summary", "output_json"], "STEP_BENCHMARK_RELEASE_DECISION_OUTPUT_JSON": ["steps", "benchmark_release_decision", "output_json"], "STEP_BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_JSON": ["steps", "benchmark_engineering_signals", "output_json"], "STEP_BENCHMARK_REALDATA_SIGNALS_OUTPUT_JSON": ["steps", "benchmark_realdata_signals", "output_json"], "STEP_BENCHMARK_REALDATA_SCORECARD_OUTPUT_JSON": ["steps", "benchmark_realdata_scorecard", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON": ["steps", "benchmark_knowledge_readiness", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_JSON": ["steps", "benchmark_knowledge_drift", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_JSON": ["steps", "benchmark_knowledge_application", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_OUTPUT_JSON": ["steps", "benchmark_knowledge_realdata_correlation", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_matrix", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_capability_matrix", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_api_surface_matrix", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_surface_action_plan", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_capability_drift", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_control_plane_drift", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_release_surface_alignment", "output_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_domain_release_gate_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_release_gate", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_action_plan", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_OUTPUT_JSON": ["steps", "benchmark_knowledge_domain_control_plane", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_JSON": ["steps", "benchmark_knowledge_reference_inventory", "output_json"], "WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_JSON": ["inputs", "benchmark_artifact_bundle_knowledge_reference_inventory_json"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_OUTPUT_JSON": ["steps", "benchmark_knowledge_source_coverage", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_JSON": ["steps", "benchmark_knowledge_source_action_plan", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_OUTPUT_JSON": ["steps", "benchmark_knowledge_outcome_correlation", "output_json"], "STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_OUTPUT_JSON": ["steps", "benchmark_knowledge_outcome_drift", "output_json"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_JSON": ["steps", "benchmark_competitive_surpass_index", "output_json"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_OUTPUT_JSON": ["steps", "benchmark_competitive_surpass_trend", "output_json"], "STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_OUTPUT_JSON": ["steps", "benchmark_competitive_surpass_action_plan", "output_json"]}
for env_name, spec in mapping.items():
    value = ''
    if spec[0] == 'inputs':
        value = str(inputs.get(spec[1]) or '')
    elif spec[0] == 'steps':
        value = step_output(steps, spec[1], spec[2])
    print(f'export {env_name}={shlex.quote(value)}')
INNER
)"

ENABLE_OVERRIDE="${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ENABLE:-}"
ENABLE_VALUE="${BENCHMARK_ARTIFACT_BUNDLE_ENABLE:-false}"
if [[ -n "$ENABLE_OVERRIDE" ]]; then
  ENABLE_VALUE="$ENABLE_OVERRIDE"
fi
ENABLE_TOKEN="$(echo "$ENABLE_VALUE" | tr '[:upper:]' '[:lower:]' | xargs)"

OUTPUT_JSON="${BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_JSON:-reports/history_sequence_eval/benchmark_artifact_bundle.json}"
OUTPUT_MD="${BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_MD:-reports/history_sequence_eval/benchmark_artifact_bundle.md}"
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$(dirname "$OUTPUT_MD")"

CMD=(
  python3 scripts/export_benchmark_artifact_bundle.py
  --title "${BENCHMARK_ARTIFACT_BUNDLE_TITLE:-CAD Benchmark Artifact Bundle}"
  --output-json "$OUTPUT_JSON"
  --output-md "$OUTPUT_MD"
)
INPUT_COUNT=0

add_if_exists() {
  local flag="$1"
  local candidate="$2"
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    CMD+=("$flag" "$candidate")
    INPUT_COUNT=$((INPUT_COUNT + 1))
  fi
}

add_if_exists --benchmark-scorecard "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_JSON:-}"
add_if_exists --benchmark-operational-summary "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_SUMMARY_JSON:-}"
add_if_exists --feedback-flywheel "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_JSON:-}"
add_if_exists --assistant-evidence "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_JSON:-}"
add_if_exists --review-queue "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_JSON:-}"
add_if_exists --ocr-review "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_OCR_REVIEW_JSON:-}"
add_if_exists --benchmark-companion-summary "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPANION_SUMMARY_JSON:-}"
add_if_exists --benchmark-release-decision "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_RELEASE_DECISION_JSON:-}"
add_if_exists --benchmark-engineering-signals "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_SIGNALS_JSON:-}"
add_if_exists --benchmark-realdata-signals "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SIGNALS_JSON:-}"
add_if_exists --benchmark-realdata-scorecard "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_JSON:-}"
add_if_exists --benchmark-knowledge-readiness "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_READINESS_JSON:-}"
add_if_exists --benchmark-knowledge-drift "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_JSON:-}"
add_if_exists --benchmark-knowledge-application "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_JSON:-}"
add_if_exists --benchmark-knowledge-realdata-correlation "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_JSON:-}"
add_if_exists --benchmark-knowledge-domain-matrix "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_JSON:-}"
add_if_exists --benchmark-knowledge-domain-capability-matrix "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_JSON:-}"
add_if_exists --benchmark-knowledge-domain-capability-drift "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-control-plane-drift "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-release-surface-alignment "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-action-plan "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-knowledge-source-coverage "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_JSON:-}"
add_if_exists --benchmark-knowledge-source-action-plan "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-knowledge-source-drift "${STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-correlation "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-drift "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_JSON:-}"
add_if_exists --benchmark-competitive-surpass-index "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_JSON:-}"
add_if_exists --benchmark-competitive-surpass-trend "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_JSON:-}"
add_if_exists --benchmark-competitive-surpass-action-plan "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-scorecard "${STEP_BENCHMARK_SCORECARD_OUTPUT_JSON:-}"
add_if_exists --benchmark-scorecard "${BENCHMARK_ARTIFACT_BUNDLE_SCORECARD_JSON:-}"
add_if_exists --benchmark-operational-summary "${STEP_BENCHMARK_OPERATIONAL_SUMMARY_OUTPUT_JSON:-}"
add_if_exists --benchmark-operational-summary "${BENCHMARK_ARTIFACT_BUNDLE_OPERATIONAL_SUMMARY_JSON:-}"
add_if_exists --feedback-flywheel "${STEP_FEEDBACK_FLYWHEEL_BENCHMARK_OUTPUT_JSON:-}"
add_if_exists --feedback-flywheel "${BENCHMARK_ARTIFACT_BUNDLE_FEEDBACK_JSON:-}"
add_if_exists --assistant-evidence "${STEP_ASSISTANT_EVIDENCE_REPORT_OUTPUT_JSON:-}"
add_if_exists --assistant-evidence "${BENCHMARK_ARTIFACT_BUNDLE_ASSISTANT_JSON:-}"
add_if_exists --review-queue "${STEP_ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON:-}"
add_if_exists --review-queue "${BENCHMARK_ARTIFACT_BUNDLE_REVIEW_QUEUE_JSON:-}"
add_if_exists --ocr-review "${STEP_OCR_REVIEW_PACK_OUTPUT_JSON:-}"
add_if_exists --ocr-review "${BENCHMARK_ARTIFACT_BUNDLE_OCR_REVIEW_JSON:-}"
add_if_exists --benchmark-companion-summary "${STEP_BENCHMARK_COMPANION_SUMMARY_OUTPUT_JSON:-}"
add_if_exists --benchmark-companion-summary "${BENCHMARK_ARTIFACT_BUNDLE_COMPANION_SUMMARY_JSON:-}"
add_if_exists --benchmark-release-decision "${STEP_BENCHMARK_RELEASE_DECISION_OUTPUT_JSON:-}"
add_if_exists --benchmark-release-decision "${BENCHMARK_ARTIFACT_BUNDLE_RELEASE_DECISION_JSON:-}"
add_if_exists --benchmark-engineering-signals "${STEP_BENCHMARK_ENGINEERING_SIGNALS_OUTPUT_JSON:-}"
add_if_exists --benchmark-engineering-signals "${BENCHMARK_ARTIFACT_BUNDLE_ENGINEERING_SIGNALS_JSON:-}"
add_if_exists --benchmark-realdata-signals "${STEP_BENCHMARK_REALDATA_SIGNALS_OUTPUT_JSON:-}"
add_if_exists --benchmark-realdata-signals "${BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SIGNALS_JSON:-}"
add_if_exists --benchmark-realdata-scorecard "${STEP_BENCHMARK_REALDATA_SCORECARD_OUTPUT_JSON:-}"
add_if_exists --benchmark-realdata-scorecard "${BENCHMARK_ARTIFACT_BUNDLE_REALDATA_SCORECARD_JSON:-}"
add_if_exists --benchmark-knowledge-readiness "${STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-readiness "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_READINESS_JSON:-}"
add_if_exists --benchmark-knowledge-drift "${STEP_BENCHMARK_KNOWLEDGE_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-drift "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DRIFT_JSON:-}"
add_if_exists --benchmark-knowledge-application "${STEP_BENCHMARK_KNOWLEDGE_APPLICATION_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-application "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_APPLICATION_JSON:-}"
add_if_exists --benchmark-knowledge-realdata-correlation "${STEP_BENCHMARK_KNOWLEDGE_REALDATA_CORRELATION_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-realdata-correlation "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REALDATA_CORRELATION_JSON:-}"
add_if_exists --benchmark-knowledge-domain-matrix "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_MATRIX_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-matrix "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_MATRIX_JSON:-}"
add_if_exists --benchmark-knowledge-domain-capability-matrix "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-capability-matrix "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CAPABILITY_MATRIX_JSON:-}"
add_if_exists --benchmark-knowledge-domain-api-surface-matrix "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_API_SURFACE_MATRIX_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-surface-action-plan "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-surface-action-plan "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_SURFACE_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-knowledge-domain-capability-drift "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CAPABILITY_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-control-plane-drift "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-release-surface-alignment "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_SURFACE_ALIGNMENT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-control-plane-drift "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_DRIFT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-release-gate "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_JSON:-}"
add_if_exists --benchmark-knowledge-domain-release-gate "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_RELEASE_GATE_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-release-gate "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_RELEASE_GATE_JSON:-}"
add_if_exists --benchmark-knowledge-domain-action-plan "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_ACTION_PLAN_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-action-plan "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-knowledge-domain-control-plane "${STEP_BENCHMARK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-domain-control-plane "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_JSON:-}"
add_if_exists --benchmark-knowledge-reference-inventory "${STEP_BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-reference-inventory "${WF_INPUT_BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_JSON:-}"
add_if_exists --benchmark-knowledge-reference-inventory "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_REFERENCE_INVENTORY_JSON:-}"
add_if_exists --benchmark-knowledge-source-coverage "${STEP_BENCHMARK_KNOWLEDGE_SOURCE_COVERAGE_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-source-coverage "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_COVERAGE_JSON:-}"
add_if_exists --benchmark-knowledge-source-action-plan "${STEP_BENCHMARK_KNOWLEDGE_SOURCE_ACTION_PLAN_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-source-action-plan "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_SOURCE_ACTION_PLAN_JSON:-}"
add_if_exists --benchmark-knowledge-source-drift "${STEP_BENCHMARK_KNOWLEDGE_SOURCE_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-correlation "${STEP_BENCHMARK_KNOWLEDGE_OUTCOME_CORRELATION_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-correlation "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_CORRELATION_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-drift "${STEP_BENCHMARK_KNOWLEDGE_OUTCOME_DRIFT_OUTPUT_JSON:-}"
add_if_exists --benchmark-knowledge-outcome-drift "${BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_OUTCOME_DRIFT_JSON:-}"
add_if_exists --benchmark-competitive-surpass-index "${STEP_BENCHMARK_COMPETITIVE_SURPASS_INDEX_OUTPUT_JSON:-}"
add_if_exists --benchmark-competitive-surpass-index "${BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_INDEX_JSON:-}"
add_if_exists --benchmark-competitive-surpass-trend "${STEP_BENCHMARK_COMPETITIVE_SURPASS_TREND_OUTPUT_JSON:-}"
add_if_exists --benchmark-competitive-surpass-trend "${BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_TREND_JSON:-}"
add_if_exists --benchmark-competitive-surpass-action-plan "${STEP_BENCHMARK_COMPETITIVE_SURPASS_ACTION_PLAN_OUTPUT_JSON:-}"
add_if_exists --benchmark-competitive-surpass-action-plan "${BENCHMARK_ARTIFACT_BUNDLE_COMPETITIVE_SURPASS_ACTION_PLAN_JSON:-}"

if [[ "$ENABLE_TOKEN" != "1" && "$ENABLE_TOKEN" != "true" && "$ENABLE_TOKEN" != "yes" && "$ENABLE_TOKEN" != "on" ]]; then
  if [[ "$INPUT_COUNT" -eq 0 ]]; then
    echo "enabled=false" >> $GITHUB_OUTPUT
    echo "BENCHMARK_ARTIFACT_BUNDLE_ENABLE is not true and no inputs found; skip."
    exit 0
  fi
fi

if [[ "$INPUT_COUNT" -eq 0 ]]; then
  echo "enabled=false" >> $GITHUB_OUTPUT
  echo "No benchmark artifact bundle inputs found; skip."
  exit 0
fi

"${CMD[@]}"

echo "enabled=true" >> $GITHUB_OUTPUT
echo "output_json=$OUTPUT_JSON" >> $GITHUB_OUTPUT
echo "output_md=$OUTPUT_MD" >> $GITHUB_OUTPUT
python3 - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    Path(os.environ["BENCHMARK_ARTIFACT_BUNDLE_OUTPUT_JSON"]).read_text(
        encoding="utf-8"
    )
)
components = payload.get("component_statuses") or {}
blockers = payload.get("blockers") or []
recommendations = payload.get("recommendations") or []
drift = payload.get("knowledge_drift") or {}
drift_recommendations = payload.get("knowledge_drift_recommendations") or []
drift_changes = payload.get("knowledge_drift_component_changes") or []
operator_drift = payload.get("operator_adoption_knowledge_drift") or {}
operator_outcome_drift = (
    payload.get("operator_adoption_knowledge_outcome_drift") or {}
)
drift_domain_regressions = drift.get("domain_regressions") or []
drift_domain_improvements = drift.get("domain_improvements") or []
drift_resolved_domains = drift.get("resolved_priority_domains") or []
drift_new_domains = drift.get("new_priority_domains") or []

def compact(items):
    return "; ".join(str(item).strip() for item in items[:3] if str(item).strip())

with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
    f.write(f"overall_status={payload.get('overall_status', 'unknown')}\n")
    f.write(
        f"available_artifact_count={payload.get('available_artifact_count', 0)}\n"
    )
    f.write(
        f"feedback_status={components.get('feedback_flywheel', 'unknown')}\n"
    )
    f.write(
        f"assistant_status={components.get('assistant_explainability', 'unknown')}\n"
    )
    f.write(f"review_queue_status={components.get('review_queue', 'unknown')}\n")
    f.write(f"ocr_status={components.get('ocr_review', 'unknown')}\n")
    f.write(
        "knowledge_status="
        f"{components.get('knowledge_readiness', 'unknown')}\n"
    )
    f.write(
        "engineering_status="
        f"{components.get('engineering_signals', 'unknown')}\n"
    )
    f.write(
        f"realdata_status={components.get('realdata_signals', 'unknown')}\n"
    )
    f.write(f"knowledge_drift_status={drift.get('status', 'unknown')}\n")
    f.write(
        "knowledge_drift_summary="
        f"{compact([payload.get('knowledge_drift_summary')])}\n"
    )
    f.write(
        "knowledge_drift_recommendations="
        f"{compact(drift_recommendations)}\n"
    )
    f.write(
        "knowledge_drift_component_changes="
        + compact(
            [
                f"{row.get('component')}:{row.get('direction')}"
                for row in drift_changes
                if row.get("component")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_drift_domain_regressions="
        f"{compact(drift_domain_regressions)}\n"
    )
    f.write(
        "knowledge_drift_domain_improvements="
        f"{compact(drift_domain_improvements)}\n"
    )
    f.write(
        "knowledge_drift_resolved_priority_domains="
        f"{compact(drift_resolved_domains)}\n"
    )
    f.write(
        "knowledge_drift_new_priority_domains="
        f"{compact(drift_new_domains)}\n"
    )
    f.write(
        "operator_adoption_knowledge_drift_status="
        f"{operator_drift.get('status', 'unknown')}\n"
    )
    f.write(
        "operator_adoption_knowledge_drift_summary="
        f"{compact([operator_drift.get('summary')])}\n"
    )
    scorecard_operator = payload.get("scorecard_operator_adoption") or {}
    operational_operator = payload.get("operational_operator_adoption") or {}
    release_alignment = (
        payload.get("operator_adoption_release_surface_alignment") or {}
    )
    knowledge_domain_release_gate = (
        payload.get("knowledge_domain_release_gate") or {}
    )
    knowledge_domain_release_alignment = (
        payload.get("knowledge_domain_release_surface_alignment") or {}
    )
    f.write(
        "scorecard_operator_adoption_status="
        f"{scorecard_operator.get('status', 'unknown')}\n"
    )
    f.write(
        "scorecard_operator_adoption_mode="
        f"{scorecard_operator.get('operator_mode', 'unknown')}\n"
    )
    f.write(
        "scorecard_operator_adoption_knowledge_outcome_drift_status="
        f"{scorecard_operator.get('knowledge_outcome_drift_status', 'unknown')}\n"
    )
    f.write(
        "scorecard_operator_adoption_knowledge_outcome_drift_summary="
        f"{compact([scorecard_operator.get('knowledge_outcome_drift_summary')])}\n"
    )
    f.write(
        "operational_operator_adoption_status="
        f"{operational_operator.get('status', 'unknown')}\n"
    )
    f.write(
        "operational_operator_adoption_knowledge_outcome_drift_status="
        f"{operational_operator.get('knowledge_outcome_drift_status', 'unknown')}\n"
    )
    f.write(
        "operational_operator_adoption_knowledge_outcome_drift_summary="
        f"{compact([operational_operator.get('knowledge_outcome_drift_summary')])}\n"
    )
    f.write(
        "operator_adoption_release_surface_alignment_status="
        f"{release_alignment.get('status', 'unknown')}\n"
    )
    f.write(
        "operator_adoption_release_surface_alignment_summary="
        f"{compact([release_alignment.get('summary')])}\n"
    )
    f.write(
        "operator_adoption_release_surface_alignment_mismatches="
        f"{compact(release_alignment.get('mismatches') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_surface_alignment_status="
        f"{knowledge_domain_release_alignment.get('status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_release_surface_alignment_summary="
        f"{compact([knowledge_domain_release_alignment.get('summary')])}\n"
    )
    f.write(
        "knowledge_domain_release_surface_alignment_mismatches="
        f"{compact(knowledge_domain_release_alignment.get('mismatches') or [])}\n"
    )
    f.write(
        "operator_adoption_knowledge_outcome_drift_status="
        f"{operator_outcome_drift.get('status', 'unknown')}\n"
    )
    f.write(
        "operator_adoption_knowledge_outcome_drift_summary="
        f"{compact([operator_outcome_drift.get('summary')])}\n"
    )
    f.write(
        "realdata_scorecard_status="
        f"{payload.get('realdata_scorecard_status', 'unknown')}\n"
    )
    f.write(
        "realdata_scorecard_recommendations="
        f"{compact(payload.get('realdata_scorecard_recommendations') or [])}\n"
    )
    f.write(
        "realdata_recommendations="
        f"{compact(payload.get('realdata_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_application_status="
        f"{payload.get('knowledge_application_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_application_focus_areas="
        + compact(
            [
                f"{row.get('domain')}:{row.get('priority')}"
                for row in (
                    payload.get('knowledge_application_focus_areas') or []
                )
                if row.get("domain")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_application_priority_domains="
        f"{compact(payload.get('knowledge_application_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_application_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get('knowledge_application_domains') or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_application_recommendations="
        f"{compact(payload.get('knowledge_application_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_capability_matrix_status="
        f"{payload.get('knowledge_domain_capability_matrix_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_capability_matrix_focus_areas="
        + compact(
            [
                f"{row.get('domain')}:{row.get('priority')}"
                for row in (
                    payload.get('knowledge_domain_capability_matrix_focus_areas_detail')
                    or []
                )
                if row.get("domain")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_capability_matrix_priority_domains="
        f"{compact(payload.get('knowledge_domain_capability_matrix_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_capability_matrix_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get('knowledge_domain_capability_matrix_domains') or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_capability_matrix_recommendations="
        f"{compact(payload.get('knowledge_domain_capability_matrix_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_capability_drift_status="
        f"{payload.get('knowledge_domain_capability_drift_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_capability_drift_domain_regressions="
        f"{compact(payload.get('knowledge_domain_capability_drift_domain_regressions') or [])}\n"
    )
    f.write(
        "knowledge_domain_capability_drift_domain_improvements="
        f"{compact(payload.get('knowledge_domain_capability_drift_domain_improvements') or [])}\n"
    )
    f.write(
        "knowledge_domain_capability_drift_recommendations="
        f"{compact(payload.get('knowledge_domain_capability_drift_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_status="
        f"{payload.get('knowledge_domain_control_plane_drift_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_domain_regressions="
        f"{compact(payload.get('knowledge_domain_control_plane_drift_domain_regressions') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_domain_improvements="
        f"{compact(payload.get('knowledge_domain_control_plane_drift_domain_improvements') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_resolved_release_blockers="
        f"{compact(payload.get('knowledge_domain_control_plane_drift_resolved_release_blockers') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_new_release_blockers="
        f"{compact(payload.get('knowledge_domain_control_plane_drift_new_release_blockers') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_drift_recommendations="
        f"{compact(payload.get('knowledge_domain_control_plane_drift_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_action_plan_status="
        f"{payload.get('knowledge_domain_action_plan_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_action_plan_actions="
        + compact(
            [
                row.get("id")
                for row in (
                    payload.get("knowledge_domain_action_plan_actions") or []
                )
                if row.get("id")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_action_plan_priority_domains="
        f"{compact(payload.get('knowledge_domain_action_plan_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_action_plan_recommendations="
        f"{compact(payload.get('knowledge_domain_action_plan_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_status="
        f"{payload.get('knowledge_domain_control_plane_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_control_plane_domains="
        f"{compact((payload.get('knowledge_domain_control_plane_domains') or {}).keys())}\n"
    )
    f.write(
        "knowledge_domain_control_plane_release_blockers="
        f"{compact(payload.get('knowledge_domain_control_plane_release_blockers') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_recommendations="
        f"{compact(payload.get('knowledge_domain_control_plane_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_status="
        f"{payload.get('knowledge_domain_release_gate_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_release_gate_summary="
        f"{compact([payload.get('knowledge_domain_release_gate_summary')])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_gate_open="
        f"{str(bool(payload.get('knowledge_domain_release_gate_gate_open'))).lower()}\n"
    )
    f.write(
        "knowledge_domain_release_gate_blocking_reasons="
        f"{compact(payload.get('knowledge_domain_release_gate_blocking_reasons') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_releasable_domains="
        f"{compact(payload.get('knowledge_domain_release_gate_releasable_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_blocked_domains="
        f"{compact(payload.get('knowledge_domain_release_gate_blocked_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_priority_domains="
        f"{compact(payload.get('knowledge_domain_release_gate_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_release_gate_recommendations="
        f"{compact(payload.get('knowledge_domain_release_gate_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_reference_inventory_status="
        f"{payload.get('knowledge_reference_inventory_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_reference_inventory_priority_domains="
        f"{compact(payload.get('knowledge_reference_inventory_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_reference_inventory_total_reference_items="
        f"{payload.get('knowledge_reference_inventory_total_reference_items', 0)}\n"
    )
    f.write(
        "knowledge_source_coverage_status="
        f"{payload.get('knowledge_source_coverage_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_coverage_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get("knowledge_source_coverage_domains") or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_coverage_expansion_candidates="
        + compact(
            [
                row.get("name")
                for row in (
                    payload.get("knowledge_source_coverage_expansion_candidates")
                    or []
                )
                if row.get("name")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_coverage_recommendations="
        f"{compact(payload.get('knowledge_source_coverage_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_source_action_plan_status="
        f"{payload.get('knowledge_source_action_plan_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_action_plan_priority_domains="
        f"{compact(payload.get('knowledge_source_action_plan_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_action_plan_recommended_first_actions="
        + compact(
            [
                row.get("id")
                for row in (
                    payload.get("knowledge_source_action_plan_recommended_first_actions")
                    or []
                )
                if row.get("id")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_action_plan_source_group_action_counts="
        + compact(
            [
                f"{name}:{count}"
                for name, count in (
                    payload.get(
                        "knowledge_source_action_plan_source_group_action_counts"
                    )
                    or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_action_plan_recommendations="
        f"{compact(payload.get('knowledge_source_action_plan_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_status="
        f"{payload.get('knowledge_source_drift_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_drift_summary="
        f"{compact([payload.get('knowledge_source_drift_summary')])}\n"
    )
    f.write(
        "knowledge_source_drift_source_group_regressions="
        f"{compact(payload.get('knowledge_source_drift_source_group_regressions') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_source_group_improvements="
        f"{compact(payload.get('knowledge_source_drift_source_group_improvements') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_resolved_priority_domains="
        f"{compact(payload.get('knowledge_source_drift_resolved_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_new_priority_domains="
        f"{compact(payload.get('knowledge_source_drift_new_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_recommendations="
        f"{compact(payload.get('knowledge_source_drift_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_realdata_correlation_status="
        f"{payload.get('knowledge_realdata_correlation_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_realdata_correlation_focus_areas="
        + compact(
            [
                f"{row.get('domain')}:{row.get('priority')}"
                for row in (
                    payload.get('knowledge_realdata_correlation_focus_areas')
                    or payload.get('knowledge_realdata_correlation', {}).get(
                        'focus_areas_detail',
                        [],
                    )
                )
                if row.get("domain")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_realdata_correlation_priority_domains="
        f"{compact(payload.get('knowledge_realdata_correlation_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_realdata_correlation_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get('knowledge_realdata_correlation_domains') or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_realdata_correlation_recommendations="
        f"{compact(payload.get('knowledge_realdata_correlation_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_matrix_status="
        f"{payload.get('knowledge_domain_matrix_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_matrix_focus_areas="
        + compact(
            [
                f"{row.get('domain')}:{row.get('priority')}"
                for row in (
                    payload.get('knowledge_domain_matrix', {}).get(
                        'focus_areas_detail',
                        [],
                    )
                )
                if row.get("domain")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_matrix_priority_domains="
        f"{compact(payload.get('knowledge_domain_matrix_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_matrix_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get('knowledge_domain_matrix_domains') or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_matrix_recommendations="
        f"{compact(payload.get('knowledge_domain_matrix_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_action_plan_status="
        f"{payload.get('knowledge_domain_action_plan_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_action_plan_actions="
        + compact(
            [
                row.get("id")
                for row in (
                    payload.get("knowledge_domain_action_plan_actions") or []
                )
                if row.get("id")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_domain_action_plan_priority_domains="
        f"{compact(payload.get('knowledge_domain_action_plan_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_domain_action_plan_recommendations="
        f"{compact(payload.get('knowledge_domain_action_plan_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_status="
        f"{payload.get('knowledge_domain_control_plane_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_domain_control_plane_domains="
        f"{compact((payload.get('knowledge_domain_control_plane_domains') or {}).keys())}\n"
    )
    f.write(
        "knowledge_domain_control_plane_release_blockers="
        f"{compact(payload.get('knowledge_domain_control_plane_release_blockers') or [])}\n"
    )
    f.write(
        "knowledge_domain_control_plane_recommendations="
        f"{compact(payload.get('knowledge_domain_control_plane_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_source_coverage_status="
        f"{payload.get('knowledge_source_coverage_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_coverage_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get("knowledge_source_coverage_domains") or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_coverage_expansion_candidates="
        + compact(
            [
                row.get("name")
                for row in (
                    payload.get("knowledge_source_coverage_expansion_candidates")
                    or []
                )
                if row.get("name")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_coverage_recommendations="
        f"{compact(payload.get('knowledge_source_coverage_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_source_action_plan_status="
        f"{payload.get('knowledge_source_action_plan_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_action_plan_priority_domains="
        f"{compact(payload.get('knowledge_source_action_plan_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_action_plan_recommended_first_actions="
        + compact(
            [
                row.get("id")
                for row in (
                    payload.get("knowledge_source_action_plan_recommended_first_actions")
                    or []
                )
                if row.get("id")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_action_plan_source_group_action_counts="
        + compact(
            [
                f"{name}:{count}"
                for name, count in (
                    payload.get(
                        "knowledge_source_action_plan_source_group_action_counts"
                    )
                    or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_source_action_plan_recommendations="
        f"{compact(payload.get('knowledge_source_action_plan_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_status="
        f"{payload.get('knowledge_source_drift_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_source_drift_summary="
        f"{compact([payload.get('knowledge_source_drift_summary')])}\n"
    )
    f.write(
        "knowledge_source_drift_source_group_regressions="
        f"{compact(payload.get('knowledge_source_drift_source_group_regressions') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_source_group_improvements="
        f"{compact(payload.get('knowledge_source_drift_source_group_improvements') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_resolved_priority_domains="
        f"{compact(payload.get('knowledge_source_drift_resolved_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_new_priority_domains="
        f"{compact(payload.get('knowledge_source_drift_new_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_source_drift_recommendations="
        f"{compact(payload.get('knowledge_source_drift_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_outcome_correlation_status="
        f"{payload.get('knowledge_outcome_correlation_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_outcome_correlation_focus_areas="
        + compact(
            [
                f"{row.get('domain')}:{row.get('priority')}"
                for row in (
                    payload.get('knowledge_outcome_correlation', {}).get(
                        'focus_areas_detail',
                        [],
                    )
                )
                if row.get("domain")
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_outcome_correlation_priority_domains="
        f"{compact(payload.get('knowledge_outcome_correlation_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_outcome_correlation_domain_statuses="
        + compact(
            [
                f"{name}:{row.get('status')}"
                for name, row in (
                    payload.get('knowledge_outcome_correlation_domains') or {}
                ).items()
                if name
            ]
        )
        + "\n"
    )
    f.write(
        "knowledge_outcome_correlation_recommendations="
        f"{compact(payload.get('knowledge_outcome_correlation_recommendations') or [])}\n"
    )
    f.write(
        "knowledge_outcome_drift_status="
        f"{payload.get('knowledge_outcome_drift_status', 'unknown')}\n"
    )
    f.write(
        "knowledge_outcome_drift_summary="
        f"{compact([payload.get('knowledge_outcome_drift_summary')])}\n"
    )
    f.write(
        "knowledge_outcome_drift_domain_regressions="
        f"{compact(payload.get('knowledge_outcome_drift_domain_regressions') or [])}\n"
    )
    f.write(
        "knowledge_outcome_drift_domain_improvements="
        f"{compact(payload.get('knowledge_outcome_drift_domain_improvements') or [])}\n"
    )
    f.write(
        "knowledge_outcome_drift_resolved_priority_domains="
        f"{compact(payload.get('knowledge_outcome_drift_resolved_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_outcome_drift_new_priority_domains="
        f"{compact(payload.get('knowledge_outcome_drift_new_priority_domains') or [])}\n"
    )
    f.write(
        "knowledge_outcome_drift_recommendations="
        f"{compact(payload.get('knowledge_outcome_drift_recommendations') or [])}\n"
    )
    f.write(
        "competitive_surpass_index_status="
        f"{payload.get('competitive_surpass_index_status', 'unknown')}\n"
    )
    f.write(
        "competitive_surpass_primary_gaps="
        f"{compact(payload.get('competitive_surpass_primary_gaps') or [])}\n"
    )
    f.write(
        "competitive_surpass_recommendations="
        f"{compact(payload.get('competitive_surpass_recommendations') or [])}\n"
    )
    f.write(
        "competitive_surpass_trend_status="
        f"{payload.get('competitive_surpass_trend_status', 'unknown')}\n"
    )
    f.write(
        "competitive_surpass_trend_summary="
        f"{compact([payload.get('competitive_surpass_trend_summary')])}\n"
    )
    f.write(
        "competitive_surpass_trend_recommendations="
        f"{compact(payload.get('competitive_surpass_trend_recommendations') or [])}\n"
    )
    f.write(
        "competitive_surpass_action_plan_status="
        f"{payload.get('competitive_surpass_action_plan_status', 'unknown')}\n"
    )
    f.write(
        "competitive_surpass_action_plan_total_action_count="
        f"{payload.get('competitive_surpass_action_plan_total_action_count', 0)}\n"
    )
    f.write(
        "competitive_surpass_action_plan_priority_pillars="
        f"{compact(payload.get('competitive_surpass_action_plan_priority_pillars') or [])}\n"
    )
    f.write(
        "competitive_surpass_action_plan_recommendations="
        f"{compact(payload.get('competitive_surpass_action_plan_recommendations') or [])}\n"
    )
    f.write(f"blockers={compact(blockers)}\n")
    f.write(f"recommendations={compact(recommendations)}\n")
PY
