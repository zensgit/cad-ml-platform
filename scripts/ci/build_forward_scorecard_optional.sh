#!/usr/bin/env bash
set -euo pipefail

eval "$(python3 - <<'INNER'
import json
import os
import shlex


def load_json_env(name):
    try:
        payload = json.loads(os.environ.get(name) or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def step_output(steps, step_id, output_name):
    step = steps.get(step_id) or {}
    outputs = step.get("outputs") or {}
    value = outputs.get(output_name)
    if value is None:
        return ""
    return str(value)


inputs = load_json_env("GITHUB_EVENT_INPUTS_JSON")
steps = load_json_env("GITHUB_STEPS_JSON")
mapping = {
    "WF_INPUT_FORWARD_SCORECARD_ENABLE": ["inputs", "forward_scorecard_enable"],
    "WF_INPUT_FORWARD_SCORECARD_MODEL_READINESS_JSON": [
        "inputs",
        "forward_scorecard_model_readiness_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_HYBRID_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_hybrid_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_graph2d_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_HISTORY_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_history_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_BREP_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_brep_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_QDRANT_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_qdrant_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_review_queue_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_knowledge_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON": [
        "inputs",
        "forward_scorecard_manufacturing_evidence_summary_json",
    ],
    "WF_INPUT_FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV": [
        "inputs",
        "forward_scorecard_manufacturing_review_manifest_csv",
    ],
    "STEP_GRAPH2D_BLIND_GATE_SUMMARY_PATH": [
        "steps",
        "graph2d_blind_gate",
        "summary_path",
    ],
    "STEP_ACTIVE_LEARNING_REVIEW_QUEUE_OUTPUT_JSON": [
        "steps",
        "active_learning_review_queue_report",
        "output_json",
    ],
    "STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON": [
        "steps",
        "benchmark_knowledge_readiness",
        "output_json",
    ],
    "STEP_BREP_GOLDEN_EVAL_SUMMARY_JSON": [
        "steps",
        "brep_golden_manifest",
        "eval_summary_json",
    ],
}

for env_name, spec in mapping.items():
    value = ""
    if spec[0] == "inputs":
        value = str(inputs.get(spec[1]) or "")
    elif spec[0] == "steps":
        value = step_output(steps, spec[1], spec[2])
    print(f"export {env_name}={shlex.quote(value)}")
INNER
)"

write_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >> "$GITHUB_OUTPUT"
  fi
}

is_true() {
  local token
  token="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]' | xargs)"
  [[ "$token" == "1" || "$token" == "true" || "$token" == "yes" || "$token" == "on" ]]
}

ENABLE_OVERRIDE="${WF_INPUT_FORWARD_SCORECARD_ENABLE:-}"
ENABLE_VALUE="${FORWARD_SCORECARD_ENABLE:-false}"
if [[ -n "$ENABLE_OVERRIDE" ]]; then
  ENABLE_VALUE="$ENABLE_OVERRIDE"
fi
GATE_ENABLE_VALUE="${FORWARD_SCORECARD_RELEASE_GATE_ENABLE:-false}"
GATE_REQUIRE_VALUE="${FORWARD_SCORECARD_RELEASE_GATE_REQUIRE_RELEASE:-false}"

OUTPUT_JSON="${FORWARD_SCORECARD_OUTPUT_JSON:-reports/benchmark/forward_scorecard/latest.json}"
OUTPUT_MD="${FORWARD_SCORECARD_OUTPUT_MD:-reports/benchmark/forward_scorecard/latest.md}"
GATE_OUTPUT_JSON="${FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON:-reports/benchmark/forward_scorecard/release_gate.json}"
MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json}"
MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md}"
MANUFACTURING_REVIEW_MANIFEST_GAP_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_GAP_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_gaps.csv}"
MANUFACTURING_REVIEW_CONTEXT_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_CONTEXT_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_context.csv}"
MANUFACTURING_REVIEW_BATCH_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_batch.csv}"
MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_batch_template.csv}"
MANUFACTURING_REVIEW_ASSIGNMENT_MD="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_ASSIGNMENT_MD:-reports/benchmark/forward_scorecard/manufacturing_review_assignment.md}"
MANUFACTURING_REVIEWER_TEMPLATE_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_CSV:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template.csv}"
MANUFACTURING_REVIEW_HANDOFF_MD="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD:-reports/benchmark/forward_scorecard/manufacturing_review_handoff.md}"
MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV:-}"
MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.json}"
MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.md}"
MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight_gaps.csv}"
MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS:-${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES:-30}}"
MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_manifest.applied.csv}"
MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply.json}"
MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV:-reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply_audit.csv}"
MANUFACTURING_REVIEW_BASE_MANIFEST_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV:-}"
MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_merged.csv}"
MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_merge.json}"
MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV="${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV:-reports/benchmark/forward_scorecard/manufacturing_review_manifest_merge_audit.csv}"
mkdir -p "$(dirname "$OUTPUT_JSON")"
mkdir -p "$(dirname "$OUTPUT_MD")"
mkdir -p "$(dirname "$GATE_OUTPUT_JSON")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MANIFEST_GAP_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_CONTEXT_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_BATCH_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_ASSIGNMENT_MD")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_HANDOFF_MD")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON")"
mkdir -p "$(dirname "$MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV")"
MANUFACTURING_EVIDENCE_INPUT_JSON=""
MANUFACTURING_REVIEW_MANIFEST_CSV=""
MANUFACTURING_REVIEW_MANIFEST_STATUS="missing"
MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS="missing"
MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS="missing"
MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS="missing"

CMD=(
  python3 scripts/export_forward_scorecard.py
  --title "${FORWARD_SCORECARD_TITLE:-CAD ML Forward Scorecard}"
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

add_manufacturing_evidence_if_exists() {
  local candidate="$1"
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    CMD+=(--manufacturing-evidence-summary "$candidate")
    INPUT_COUNT=$((INPUT_COUNT + 1))
    MANUFACTURING_EVIDENCE_INPUT_JSON="$candidate"
  fi
}

validate_manufacturing_review_manifest_if_exists() {
  local candidate="$1"
  if [[ -n "$candidate" && -f "$candidate" ]]; then
    if [[ -z "$MANUFACTURING_REVIEW_MANIFEST_CSV" ]]; then
      local validation_candidate="$candidate"
      if [[ -n "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" ]]; then
        if [[ -f "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" ]]; then
          MANUFACTURING_REVIEW_PREFLIGHT_CMD=(
            python3 scripts/build_manufacturing_review_manifest.py
            --validate-reviewer-template "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" \
            --summary-json "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON" \
            --reviewer-template-preflight-md "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD" \
            --reviewer-template-preflight-gap-csv "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV" \
            --min-reviewed-samples "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS" \
            --base-manifest "$candidate"
          )
          if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA:-false}"; then
            MANUFACTURING_REVIEW_PREFLIGHT_CMD+=(--require-reviewer-metadata)
          fi
          "${MANUFACTURING_REVIEW_PREFLIGHT_CMD[@]}"
          MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS="$(
            python3 - "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("status") or "unknown"))
PY
          )"
          if [[ "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS" == "ready" ]]; then
            MANUFACTURING_REVIEW_APPLY_CMD=(
              python3 scripts/build_manufacturing_review_manifest.py
            --apply-reviewer-template "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" \
            --base-manifest "$candidate" \
            --output-csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV" \
            --summary-json "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON" \
            --reviewer-template-apply-audit-csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV" \
            --min-reviewed-samples "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES:-30}"
            )
            if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA:-false}"; then
              MANUFACTURING_REVIEW_APPLY_CMD+=(--require-reviewer-metadata)
            fi
            "${MANUFACTURING_REVIEW_APPLY_CMD[@]}"
            validation_candidate="$MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV"
            MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS="$(
              python3 - "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("status") or "unknown"))
PY
            )"
          else
            MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS="blocked_preflight"
          fi
        else
          MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS="missing_template"
          MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS="missing_template"
        fi
      fi
      MANUFACTURING_REVIEW_MANIFEST_CSV="$validation_candidate"
      INPUT_COUNT=$((INPUT_COUNT + 1))
      MANUFACTURING_REVIEW_CMD=(
        python3 scripts/build_manufacturing_review_manifest.py
        --validate-manifest "$validation_candidate" \
        --summary-json "$MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON" \
        --progress-md "$MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD" \
        --gap-csv "$MANUFACTURING_REVIEW_MANIFEST_GAP_CSV" \
        --review-context-csv "$MANUFACTURING_REVIEW_CONTEXT_CSV" \
        --review-batch-csv "$MANUFACTURING_REVIEW_BATCH_CSV" \
        --review-batch-template-csv "$MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV" \
        --assignment-md "$MANUFACTURING_REVIEW_ASSIGNMENT_MD" \
        --reviewer-template-csv "$MANUFACTURING_REVIEWER_TEMPLATE_CSV" \
        --handoff-md "$MANUFACTURING_REVIEW_HANDOFF_MD" \
        --reviewer-template-preflight-md "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD" \
        --reviewer-template-preflight-gap-csv "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV" \
        --reviewer-template-preflight-min-ready-rows "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS" \
        --min-reviewed-samples "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES:-30}"
      )
      if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA:-false}"; then
        MANUFACTURING_REVIEW_CMD+=(--require-reviewer-metadata)
      fi
      "${MANUFACTURING_REVIEW_CMD[@]}"
      CMD+=(
        --manufacturing-review-manifest-validation-summary
        "$MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON"
      )
      MANUFACTURING_REVIEW_MANIFEST_STATUS="$(
        python3 - "$MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("status") or "unknown"))
PY
      )"
    fi
  fi
}

merge_manufacturing_review_manifest_if_ready() {
  if [[ -z "$MANUFACTURING_REVIEW_BASE_MANIFEST_CSV" ]]; then
    MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS="missing"
    return
  fi
  if [[ ! -f "$MANUFACTURING_REVIEW_BASE_MANIFEST_CSV" ]]; then
    MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS="missing_base_manifest"
    return
  fi
  if [[ -z "$MANUFACTURING_REVIEW_MANIFEST_CSV" ]]; then
    MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS="missing_review_manifest"
    return
  fi

  MANUFACTURING_REVIEW_MERGE_CMD=(
    python3 scripts/build_manufacturing_review_manifest.py
    --merge-approved-review-manifest "$MANUFACTURING_REVIEW_MANIFEST_CSV" \
    --base-manifest "$MANUFACTURING_REVIEW_BASE_MANIFEST_CSV" \
    --output-csv "$MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV" \
    --summary-json "$MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON" \
    --review-manifest-merge-audit-csv "$MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV"
  )
  if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA:-false}"; then
    MANUFACTURING_REVIEW_MERGE_CMD+=(--require-reviewer-metadata)
  fi
  "${MANUFACTURING_REVIEW_MERGE_CMD[@]}"
  MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS="$(
    python3 - "$MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(payload.get("status") or "unknown"))
PY
  )"
}

add_if_exists --model-readiness-summary "${WF_INPUT_FORWARD_SCORECARD_MODEL_READINESS_JSON:-}"
add_if_exists --model-readiness-summary "${FORWARD_SCORECARD_MODEL_READINESS_JSON:-}"

add_if_exists --hybrid-summary "${BENCHMARK_REALDATA_SCORECARD_HYBRID_SUMMARY_JSON:-}"
add_if_exists --hybrid-summary "${BENCHMARK_SCORECARD_HYBRID_SUMMARY_JSON:-}"
add_if_exists --hybrid-summary "${WF_INPUT_FORWARD_SCORECARD_HYBRID_SUMMARY_JSON:-}"
add_if_exists --hybrid-summary "${FORWARD_SCORECARD_HYBRID_SUMMARY_JSON:-}"

add_if_exists --graph2d-summary "${STEP_GRAPH2D_BLIND_GATE_SUMMARY_PATH:-}"
add_if_exists --graph2d-summary "${GRAPH2D_BLIND_SUMMARY_JSON:-}"
add_if_exists --graph2d-summary "${BENCHMARK_SCORECARD_GRAPH2D_BLIND_DIAGNOSE_JSON:-}"
add_if_exists --graph2d-summary "${WF_INPUT_FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON:-}"
add_if_exists --graph2d-summary "${FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON:-}"

add_if_exists --history-summary "reports/history_sequence_eval/summary.json"
add_if_exists --history-summary "${BENCHMARK_REALDATA_SCORECARD_HISTORY_SUMMARY_JSON:-}"
add_if_exists --history-summary "${BENCHMARK_SCORECARD_HISTORY_SUMMARY_JSON:-}"
add_if_exists --history-summary "${WF_INPUT_FORWARD_SCORECARD_HISTORY_SUMMARY_JSON:-}"
add_if_exists --history-summary "${FORWARD_SCORECARD_HISTORY_SUMMARY_JSON:-}"

add_if_exists --brep-summary "${BENCHMARK_REALDATA_SIGNALS_STEP_DIR_SUMMARY_JSON:-}"
add_if_exists --brep-summary "${BENCHMARK_REALDATA_SCORECARD_STEP_DIR_SUMMARY_JSON:-}"
add_if_exists --brep-summary "${STEP_BREP_GOLDEN_EVAL_SUMMARY_JSON:-}"
add_if_exists --brep-summary "${BENCHMARK_SCORECARD_BREP_SUMMARY_JSON:-}"
add_if_exists --brep-summary "${WF_INPUT_FORWARD_SCORECARD_BREP_SUMMARY_JSON:-}"
add_if_exists --brep-summary "${FORWARD_SCORECARD_BREP_SUMMARY_JSON:-}"

add_if_exists --qdrant-summary "${BENCHMARK_SCORECARD_QDRANT_READINESS_JSON:-}"
add_if_exists --qdrant-summary "${BENCHMARK_SCORECARD_QDRANT_READINESS_SUMMARY:-}"
add_if_exists --qdrant-summary "${WF_INPUT_FORWARD_SCORECARD_QDRANT_SUMMARY_JSON:-}"
add_if_exists --qdrant-summary "${FORWARD_SCORECARD_QDRANT_SUMMARY_JSON:-}"

add_if_exists --review-queue-summary "${STEP_ACTIVE_LEARNING_REVIEW_QUEUE_OUTPUT_JSON:-}"
add_if_exists --review-queue-summary "${ACTIVE_LEARNING_REVIEW_QUEUE_REPORT_OUTPUT_JSON:-}"
add_if_exists --review-queue-summary "${BENCHMARK_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON:-}"
add_if_exists --review-queue-summary "${WF_INPUT_FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON:-}"
add_if_exists --review-queue-summary "${FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON:-}"

add_if_exists --knowledge-summary "${STEP_BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON:-}"
add_if_exists --knowledge-summary "${BENCHMARK_KNOWLEDGE_READINESS_OUTPUT_JSON:-}"
add_if_exists --knowledge-summary "${BENCHMARK_SCORECARD_KNOWLEDGE_READINESS_JSON:-}"
add_if_exists --knowledge-summary "${WF_INPUT_FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON:-}"
add_if_exists --knowledge-summary "${FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON:-}"

add_manufacturing_evidence_if_exists "${BENCHMARK_SCORECARD_MANUFACTURING_EVIDENCE_JSON:-}"
add_manufacturing_evidence_if_exists "${WF_INPUT_FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON:-}"
add_manufacturing_evidence_if_exists "${FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON:-}"

validate_manufacturing_review_manifest_if_exists "${BENCHMARK_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV:-}"
validate_manufacturing_review_manifest_if_exists "${WF_INPUT_FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV:-}"
validate_manufacturing_review_manifest_if_exists "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV:-}"
merge_manufacturing_review_manifest_if_ready

if ! is_true "$ENABLE_VALUE" && ! is_true "$GATE_ENABLE_VALUE" && ! is_true "$GATE_REQUIRE_VALUE"; then
  if [[ "$INPUT_COUNT" -eq 0 ]]; then
    write_output enabled false
    write_output manufacturing_evidence_summary_available false
    write_output manufacturing_evidence_summary_json ""
    write_output manufacturing_review_manifest_available false
    write_output manufacturing_review_manifest_csv ""
    write_output manufacturing_review_manifest_summary_json ""
    write_output manufacturing_review_manifest_progress_md ""
    write_output manufacturing_review_manifest_gap_csv ""
    write_output manufacturing_review_context_csv ""
    write_output manufacturing_review_batch_csv ""
    write_output manufacturing_review_batch_template_csv ""
    write_output manufacturing_review_assignment_md ""
    write_output manufacturing_reviewer_template_csv ""
    write_output manufacturing_review_handoff_md ""
    write_output manufacturing_reviewer_template_preflight_available false
    write_output manufacturing_reviewer_template_preflight_summary_json ""
    write_output manufacturing_reviewer_template_preflight_md ""
    write_output manufacturing_reviewer_template_preflight_gap_csv ""
    write_output manufacturing_reviewer_template_preflight_status missing
    write_output manufacturing_reviewer_template_apply_available false
    write_output manufacturing_reviewer_template_apply_csv ""
    write_output manufacturing_reviewer_template_applied_manifest_csv ""
    write_output manufacturing_reviewer_template_apply_summary_json ""
    write_output manufacturing_reviewer_template_apply_audit_csv ""
    write_output manufacturing_reviewer_template_apply_status missing
    write_output manufacturing_review_manifest_status missing
    write_output manufacturing_review_manifest_merge_available false
    write_output manufacturing_review_manifest_base_csv ""
    write_output manufacturing_review_manifest_merged_csv ""
    write_output manufacturing_review_manifest_merge_summary_json ""
    write_output manufacturing_review_manifest_merge_audit_csv ""
    write_output manufacturing_review_manifest_merge_status missing
    echo "FORWARD_SCORECARD_ENABLE is not true and no inputs found; skip."
    exit 0
  fi
fi

"${CMD[@]}"

write_output enabled true
write_output output_json "$OUTPUT_JSON"
write_output output_md "$OUTPUT_MD"
write_output gate_output_json "$GATE_OUTPUT_JSON"
if [[ -n "$MANUFACTURING_EVIDENCE_INPUT_JSON" ]]; then
  write_output manufacturing_evidence_summary_available true
  write_output manufacturing_evidence_summary_json "$MANUFACTURING_EVIDENCE_INPUT_JSON"
else
  write_output manufacturing_evidence_summary_available false
  write_output manufacturing_evidence_summary_json ""
fi
if [[ -n "$MANUFACTURING_REVIEW_MANIFEST_CSV" ]]; then
  write_output manufacturing_review_manifest_available true
  write_output manufacturing_review_manifest_csv "$MANUFACTURING_REVIEW_MANIFEST_CSV"
  write_output manufacturing_review_manifest_summary_json "$MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON"
  write_output manufacturing_review_manifest_progress_md "$MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD"
  write_output manufacturing_review_manifest_gap_csv "$MANUFACTURING_REVIEW_MANIFEST_GAP_CSV"
  write_output manufacturing_review_context_csv "$MANUFACTURING_REVIEW_CONTEXT_CSV"
  write_output manufacturing_review_batch_csv "$MANUFACTURING_REVIEW_BATCH_CSV"
  write_output manufacturing_review_batch_template_csv "$MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV"
  write_output manufacturing_review_assignment_md "$MANUFACTURING_REVIEW_ASSIGNMENT_MD"
  write_output manufacturing_reviewer_template_csv "$MANUFACTURING_REVIEWER_TEMPLATE_CSV"
  write_output manufacturing_review_handoff_md "$MANUFACTURING_REVIEW_HANDOFF_MD"
  if [[ "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS" != "missing" && "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS" != "missing_template" ]]; then
    write_output manufacturing_reviewer_template_preflight_available true
    write_output manufacturing_reviewer_template_preflight_summary_json "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON"
    write_output manufacturing_reviewer_template_preflight_md "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD"
    write_output manufacturing_reviewer_template_preflight_gap_csv "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV"
  else
    write_output manufacturing_reviewer_template_preflight_available false
    write_output manufacturing_reviewer_template_preflight_summary_json ""
    write_output manufacturing_reviewer_template_preflight_md ""
    write_output manufacturing_reviewer_template_preflight_gap_csv ""
  fi
  write_output manufacturing_reviewer_template_preflight_status "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS"
  if [[ "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS" == "applied" ]]; then
    write_output manufacturing_reviewer_template_apply_available true
    write_output manufacturing_reviewer_template_apply_csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV"
    write_output manufacturing_reviewer_template_applied_manifest_csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV"
    write_output manufacturing_reviewer_template_apply_summary_json "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON"
    write_output manufacturing_reviewer_template_apply_audit_csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV"
  else
    write_output manufacturing_reviewer_template_apply_available false
    write_output manufacturing_reviewer_template_apply_csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV"
    write_output manufacturing_reviewer_template_applied_manifest_csv ""
    write_output manufacturing_reviewer_template_apply_summary_json ""
    write_output manufacturing_reviewer_template_apply_audit_csv ""
  fi
  write_output manufacturing_reviewer_template_apply_status "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS"
  write_output manufacturing_review_manifest_status "$MANUFACTURING_REVIEW_MANIFEST_STATUS"
else
  write_output manufacturing_review_manifest_available false
  write_output manufacturing_review_manifest_csv ""
  write_output manufacturing_review_manifest_summary_json ""
  write_output manufacturing_review_manifest_progress_md ""
  write_output manufacturing_review_manifest_gap_csv ""
  write_output manufacturing_review_context_csv ""
  write_output manufacturing_review_batch_csv ""
  write_output manufacturing_review_batch_template_csv ""
  write_output manufacturing_review_assignment_md ""
  write_output manufacturing_reviewer_template_csv ""
  write_output manufacturing_review_handoff_md ""
  write_output manufacturing_reviewer_template_preflight_available false
  write_output manufacturing_reviewer_template_preflight_summary_json ""
  write_output manufacturing_reviewer_template_preflight_md ""
  write_output manufacturing_reviewer_template_preflight_gap_csv ""
  write_output manufacturing_reviewer_template_preflight_status "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS"
  write_output manufacturing_reviewer_template_apply_available false
  write_output manufacturing_reviewer_template_apply_csv "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV"
  write_output manufacturing_reviewer_template_applied_manifest_csv ""
  write_output manufacturing_reviewer_template_apply_summary_json ""
  write_output manufacturing_reviewer_template_apply_audit_csv ""
  write_output manufacturing_reviewer_template_apply_status "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_STATUS"
  write_output manufacturing_review_manifest_status missing
fi
if [[ "$MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS" == "merged" ]]; then
  write_output manufacturing_review_manifest_merge_available true
  write_output manufacturing_review_manifest_base_csv "$MANUFACTURING_REVIEW_BASE_MANIFEST_CSV"
  write_output manufacturing_review_manifest_merged_csv "$MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV"
  write_output manufacturing_review_manifest_merge_summary_json "$MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON"
  write_output manufacturing_review_manifest_merge_audit_csv "$MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV"
  write_output manufacturing_review_manifest_merge_status "$MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS"
else
  write_output manufacturing_review_manifest_merge_available false
  write_output manufacturing_review_manifest_base_csv "$MANUFACTURING_REVIEW_BASE_MANIFEST_CSV"
  write_output manufacturing_review_manifest_merged_csv ""
  write_output manufacturing_review_manifest_merge_summary_json ""
  write_output manufacturing_review_manifest_merge_audit_csv ""
  write_output manufacturing_review_manifest_merge_status "$MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS"
fi

FORWARD_SCORECARD_OUTPUT_JSON="$OUTPUT_JSON" python3 - <<'PY'
import json
import os
from pathlib import Path


def compact(items):
    return "; ".join(str(item).strip() for item in (items or [])[:3] if str(item).strip())


output = os.environ.get("GITHUB_OUTPUT")
if output:
    payload = json.loads(
        Path(os.environ["FORWARD_SCORECARD_OUTPUT_JSON"]).read_text(encoding="utf-8")
    )
    components = payload.get("components") or {}
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(f"overall_status={payload.get('overall_status', 'unknown')}\n")
        handle.write(
            "model_readiness_status="
            f"{(components.get('model_readiness') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            f"hybrid_status={(components.get('hybrid_dxf') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            f"graph2d_status={(components.get('graph2d') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            "history_status="
            f"{(components.get('history_sequence') or {}).get('status', 'unknown')}\n"
        )
        handle.write(f"brep_status={(components.get('brep') or {}).get('status', 'unknown')}\n")
        handle.write(
            f"qdrant_status={(components.get('qdrant') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            "review_queue_status="
            f"{(components.get('review_queue') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            f"knowledge_status={(components.get('knowledge') or {}).get('status', 'unknown')}\n"
        )
        handle.write(
            "manufacturing_evidence_status="
            f"{(components.get('manufacturing_evidence') or {}).get('status', 'unknown')}\n"
        )
        handle.write(f"recommendations={compact(payload.get('recommendations'))}\n")
PY

if is_true "$GATE_ENABLE_VALUE" || is_true "$GATE_REQUIRE_VALUE"; then
  GATE_CMD=(
    python3 scripts/ci/check_forward_scorecard_release_gate.py
    --scorecard "$OUTPUT_JSON"
    --labels "${FORWARD_SCORECARD_RELEASE_LABELS:-}"
    --output-json "$GATE_OUTPUT_JSON"
  )
  if is_true "$GATE_REQUIRE_VALUE"; then
    GATE_CMD+=(--require-release)
  fi
  PREFIXES=()
  if [[ -n "${FORWARD_SCORECARD_RELEASE_LABEL_PREFIXES:-}" ]]; then
    IFS=', ' read -r -a PREFIXES <<< "${FORWARD_SCORECARD_RELEASE_LABEL_PREFIXES:-}"
  fi
  if [[ "${#PREFIXES[@]}" -gt 0 ]]; then
    for prefix in "${PREFIXES[@]}"; do
      if [[ -n "$prefix" ]]; then
        GATE_CMD+=(--release-label-prefix "$prefix")
      fi
    done
  fi
  "${GATE_CMD[@]}"
else
  GATE_OUTPUT_JSON="$GATE_OUTPUT_JSON" python3 - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "gate_applicable": False,
    "should_fail": False,
    "release_labels": [],
    "reason": "Forward scorecard release gate is disabled.",
}
path = Path(os.environ["GATE_OUTPUT_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  write_output gate_applicable false
  write_output should_fail false
  write_output release_labels ""
  write_output reason "Forward scorecard release gate is disabled."
fi

if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED:-false}"; then
  if [[ "$MANUFACTURING_REVIEW_MANIFEST_STATUS" != "release_label_ready" ]]; then
    echo "Manufacturing review manifest is not release-label-ready: ${MANUFACTURING_REVIEW_MANIFEST_STATUS}" >&2
    exit 1
  fi
fi

if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED:-false}"; then
  if [[ -n "$MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" && "$MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS" != "ready" ]]; then
    echo "Manufacturing reviewer template preflight is not ready: ${MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_STATUS}" >&2
    exit 1
  fi
fi

if is_true "${FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED:-false}"; then
  if [[ "$MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS" != "merged" ]]; then
    echo "Manufacturing review manifest merge is not ready: ${MANUFACTURING_REVIEW_MANIFEST_MERGE_STATUS}" >&2
    exit 1
  fi
fi
