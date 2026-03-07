#!/usr/bin/env bash
set -euo pipefail

# Save OCR golden evaluation metrics to JSON under reports/eval_history/
# Also captures git metadata for trend analysis.
# Optional: can also run history-sequence evaluation when env is configured.

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
REPORT_DIR="${EVAL_HISTORY_REPORT_DIR:-$ROOT_DIR/reports/eval_history}"
OCR_SCRIPT="${EVAL_HISTORY_OCR_SCRIPT:-$ROOT_DIR/tests/ocr/run_golden_evaluation.py}"
HISTORY_BUILD_SCRIPT="${EVAL_HISTORY_BUILD_SCRIPT:-$ROOT_DIR/scripts/build_history_sequence_prototypes.py}"
HISTORY_EVAL_SCRIPT="${EVAL_HISTORY_EVAL_SCRIPT:-$ROOT_DIR/scripts/eval_history_sequence_classifier.py}"
HISTORY_TUNE_SCRIPT="${EVAL_HISTORY_TUNE_SCRIPT:-$ROOT_DIR/scripts/tune_history_sequence_weights.py}"

mkdir -p "$REPORT_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")
branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
safe_branch=$(echo "$branch" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+|-+$//g')
safe_branch=${safe_branch:-unknown}
commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
runner="local"
[[ -n "${CI:-}" ]] && runner="ci"
machine_name=$(hostname 2>/dev/null || echo "unknown")
os_info=$(uname -s 2>/dev/null || echo "unknown")
os_version=$(uname -r 2>/dev/null || echo "unknown")
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
start_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Running OCR golden evaluation..."
ocr_tmp_out="/tmp/eval_out_${timestamp}.txt"
python3 "$OCR_SCRIPT" | tee "$ocr_tmp_out"

# Parse metrics from stdout
dimension_recall=$(grep -Eo 'dimension_recall=[0-9.]+"?' "$ocr_tmp_out" | cut -d= -f2)
brier_score=$(grep -Eo 'brier_score=[0-9.]+' "$ocr_tmp_out" | cut -d= -f2)
edge_f1=$(grep -Eo 'edge_f1=[0-9.]+' "$ocr_tmp_out" | cut -d= -f2)

outfile="$REPORT_DIR/${timestamp}_${safe_branch}_${commit}.json"

cat > "$outfile" <<JSON
{
  "schema_version": "1.0.0",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "branch": "$branch",
  "commit": "$commit",
  "type": "ocr",
  "run_context": {
    "runner": "$runner",
    "machine": "$machine_name",
    "os": "$os_info $os_version",
    "python": "$python_version",
    "start_time": "$start_time",
    "ci_job_id": "${GITHUB_RUN_ID:-null}",
    "ci_workflow": "${GITHUB_WORKFLOW:-null}"
  },
  "metrics": {
    "dimension_recall": ${dimension_recall:-0},
    "brier_score": ${brier_score:-0},
    "edge_f1": ${edge_f1:-0}
  }
}
JSON

echo "Saved eval history -> $outfile"

# Optional history-sequence evaluation
history_enable="${HISTORY_SEQUENCE_EVAL_ENABLE:-false}"
history_manifest="${HISTORY_SEQUENCE_EVAL_MANIFEST:-}"
history_h5_dir="${HISTORY_SEQUENCE_EVAL_H5_DIR:-}"
history_label_source="${HISTORY_SEQUENCE_EVAL_LABEL_SOURCE:-manifest}"
history_output_dir="${HISTORY_SEQUENCE_EVAL_OUTPUT_DIR:-$ROOT_DIR/reports/history_sequence_eval}"
history_proto_out="${HISTORY_SEQUENCE_PROTOTYPES_OUT:-$history_output_dir/prototypes.json}"
history_synonyms_path="${HISTORY_SEQUENCE_SYNONYMS_PATH:-}"
history_filename_min_conf="${HISTORY_SEQUENCE_FILENAME_MIN_CONF:-0.8}"
history_min_seq_len="${HISTORY_SEQUENCE_MIN_SEQ_LEN:-4}"
history_proto_token_weight="${HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT:-1.0}"
history_proto_bigram_weight="${HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT:-1.0}"
history_tune_enable="${HISTORY_SEQUENCE_TUNE_ENABLE:-false}"
history_tune_output_dir="${HISTORY_SEQUENCE_TUNE_OUTPUT_DIR:-$history_output_dir/tuning}"
history_tune_objective="${HISTORY_SEQUENCE_TUNE_OBJECTIVE:-macro_f1_overall}"
history_tune_token_grid="${HISTORY_SEQUENCE_TUNE_TOKEN_WEIGHT_GRID:-0.5,1.0,1.5}"
history_tune_bigram_grid="${HISTORY_SEQUENCE_TUNE_BIGRAM_WEIGHT_GRID:-0.0,0.5,1.0,1.5}"
history_tune_best_config=""
history_selected_env_file="$history_output_dir/recommended_history_sequence.env"
history_configured_token_weight="$history_proto_token_weight"
history_configured_bigram_weight="$history_proto_bigram_weight"
history_selected_source="configured"

if [[ "$history_enable" != "true" && -z "$history_manifest" && -z "$history_h5_dir" ]]; then
  echo "History-sequence evaluation skipped (not configured)."
  exit 0
fi

if [[ -n "$history_manifest" ]]; then
  history_label_source="manifest"
elif [[ -z "$history_h5_dir" ]]; then
  echo "History-sequence evaluation skipped (missing both manifest and h5 dir)."
  exit 0
fi

mkdir -p "$history_output_dir"

echo "Running history-sequence prototype build..."
build_cmd=(
  python3 "$HISTORY_BUILD_SCRIPT"
  --label-source "$history_label_source"
  --output "$history_proto_out"
)
if [[ -n "$history_manifest" ]]; then
  build_cmd+=(--manifest "$history_manifest")
fi
if [[ -n "$history_h5_dir" ]]; then
  build_cmd+=(--h5-dir "$history_h5_dir")
fi
if [[ -n "$history_synonyms_path" ]]; then
  build_cmd+=(--synonyms-path "$history_synonyms_path")
fi
build_cmd+=(--filename-min-conf "$history_filename_min_conf")

if ! "${build_cmd[@]}"; then
  echo "Warning: history prototype build failed; skipping history eval."
  exit 0
fi

if [[ "$history_tune_enable" == "true" ]]; then
  echo "Running history-sequence weight tuning..."
  tune_cmd=(
    python3 "$HISTORY_TUNE_SCRIPT"
    --label-source "$history_label_source"
    --prototypes-path "$history_proto_out"
    --output-dir "$history_tune_output_dir"
    --objective "$history_tune_objective"
    --token-weight-grid "$history_tune_token_grid"
    --bigram-weight-grid "$history_tune_bigram_grid"
    --min-seq-len "$history_min_seq_len"
    --filename-min-conf "$history_filename_min_conf"
  )
  if [[ -n "$history_manifest" ]]; then
    tune_cmd+=(--manifest "$history_manifest")
  fi
  if [[ -n "$history_h5_dir" ]]; then
    tune_cmd+=(--h5-dir "$history_h5_dir")
  fi
  if [[ -n "$history_synonyms_path" ]]; then
    tune_cmd+=(--synonyms-path "$history_synonyms_path")
  fi
  if "${tune_cmd[@]}"; then
    history_tune_best_config="$history_tune_output_dir/best_config.json"
    if [[ -f "$history_tune_best_config" ]]; then
      tuned_weights=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_tune_best_config").read_text(encoding="utf-8"))
best = payload.get("best") or {}
print(f"{best.get('token_weight', '$history_proto_token_weight')},{best.get('bigram_weight', '$history_proto_bigram_weight')}")
PY
)
      tuned_token_weight="$(echo "$tuned_weights" | cut -d, -f1 | tr -d '[:space:]')"
      tuned_bigram_weight="$(echo "$tuned_weights" | cut -d, -f2 | tr -d '[:space:]')"
      if [[ -n "$tuned_token_weight" ]]; then
        history_proto_token_weight="$tuned_token_weight"
      fi
      if [[ -n "$tuned_bigram_weight" ]]; then
        history_proto_bigram_weight="$tuned_bigram_weight"
      fi
      history_selected_source="tuned"
      echo "Tuned weights selected: token=$history_proto_token_weight, bigram=$history_proto_bigram_weight"
    fi
  else
    echo "Warning: history-sequence tuning failed; fallback to configured weights."
  fi
fi

cat > "$history_selected_env_file" <<ENV
# Auto-generated by scripts/eval_with_history.sh
# source=${history_selected_source}
HISTORY_SEQUENCE_PROTOTYPE_TOKEN_WEIGHT=${history_proto_token_weight}
HISTORY_SEQUENCE_PROTOTYPE_BIGRAM_WEIGHT=${history_proto_bigram_weight}
HISTORY_SEQUENCE_PROTOTYPES_PATH=${history_proto_out}
ENV

echo "Running history-sequence evaluation..."
eval_cmd=(
  python3 "$HISTORY_EVAL_SCRIPT"
  --label-source "$history_label_source"
  --prototypes-path "$history_proto_out"
  --output-dir "$history_output_dir"
  --min-seq-len "$history_min_seq_len"
  --filename-min-conf "$history_filename_min_conf"
  --prototype-token-weight "$history_proto_token_weight"
  --prototype-bigram-weight "$history_proto_bigram_weight"
)
if [[ -n "$history_manifest" ]]; then
  eval_cmd+=(--manifest "$history_manifest")
fi
if [[ -n "$history_h5_dir" ]]; then
  eval_cmd+=(--h5-dir "$history_h5_dir")
fi
if [[ -n "$history_synonyms_path" ]]; then
  eval_cmd+=(--synonyms-path "$history_synonyms_path")
fi

if ! "${eval_cmd[@]}"; then
  echo "Warning: history-sequence evaluation failed."
  exit 0
fi

history_summary="$history_output_dir/summary.json"
if [[ ! -f "$history_summary" ]]; then
  echo "Warning: missing history summary at $history_summary"
  exit 0
fi

history_coverage=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("coverage", 0))
PY
)
history_accuracy=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("accuracy_overall", 0))
PY
)
history_macro_f1=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("macro_f1_overall", 0))
PY
)
history_coarse_accuracy_on_ok=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("coarse_accuracy_on_ok", 0))
PY
)
history_coarse_accuracy_overall=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("coarse_accuracy_overall", 0))
PY
)
history_coarse_macro_f1_on_ok=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("coarse_macro_f1_on_ok", 0))
PY
)
history_coarse_macro_f1_overall=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(payload.get("coarse_macro_f1_overall", 0))
PY
)
history_exact_top_mismatches=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(json.dumps(payload.get("exact_top_mismatches", []), ensure_ascii=False))
PY
)
history_coarse_top_mismatches=$(python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$history_summary").read_text(encoding="utf-8"))
print(json.dumps(payload.get("coarse_top_mismatches", []), ensure_ascii=False))
PY
)

history_tune_best_config_json="null"
if [[ -n "$history_tune_best_config" && -f "$history_tune_best_config" ]]; then
  history_tune_best_config_json="\"$history_tune_best_config\""
fi

history_outfile="$REPORT_DIR/${timestamp}_${safe_branch}_${commit}_history_sequence.json"
cat > "$history_outfile" <<JSON
{
  "schema_version": "1.0.0",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "branch": "$branch",
  "commit": "$commit",
  "type": "history_sequence",
  "run_context": {
    "runner": "$runner",
    "machine": "$machine_name",
    "os": "$os_info $os_version",
    "python": "$python_version",
    "start_time": "$start_time",
    "ci_job_id": "${GITHUB_RUN_ID:-null}",
    "ci_workflow": "${GITHUB_WORKFLOW:-null}"
  },
  "metrics": {
    "coverage": ${history_coverage:-0},
    "accuracy_overall": ${history_accuracy:-0},
    "macro_f1_overall": ${history_macro_f1:-0},
    "coarse_accuracy_on_ok": ${history_coarse_accuracy_on_ok:-0},
    "coarse_accuracy_overall": ${history_coarse_accuracy_overall:-0},
    "coarse_macro_f1_on_ok": ${history_coarse_macro_f1_on_ok:-0},
    "coarse_macro_f1_overall": ${history_coarse_macro_f1_overall:-0},
    "exact_top_mismatches": ${history_exact_top_mismatches:-[]},
    "coarse_top_mismatches": ${history_coarse_top_mismatches:-[]}
  },
  "history_metrics": {
    "coverage": ${history_coverage:-0},
    "accuracy_overall": ${history_accuracy:-0},
    "macro_f1_overall": ${history_macro_f1:-0},
    "coarse_accuracy_on_ok": ${history_coarse_accuracy_on_ok:-0},
    "coarse_accuracy_overall": ${history_coarse_accuracy_overall:-0},
    "coarse_macro_f1_on_ok": ${history_coarse_macro_f1_on_ok:-0},
    "coarse_macro_f1_overall": ${history_coarse_macro_f1_overall:-0},
    "exact_top_mismatches": ${history_exact_top_mismatches:-[]},
    "coarse_top_mismatches": ${history_coarse_top_mismatches:-[]}
  },
  "artifacts": {
    "summary_json": "$history_summary",
    "results_csv": "$history_output_dir/results.csv",
    "prototypes_json": "$history_proto_out",
    "tune_best_config_json": ${history_tune_best_config_json},
    "recommended_env_file": "${history_selected_env_file:-}"
  },
  "tuning": {
    "enabled": $([[ "$history_tune_enable" == "true" ]] && echo "true" || echo "false"),
    "source": "$history_selected_source",
    "objective": "$history_tune_objective",
    "configured_token_weight": ${history_configured_token_weight:-1.0},
    "configured_bigram_weight": ${history_configured_bigram_weight:-1.0},
    "selected_token_weight": ${history_proto_token_weight:-1.0},
    "selected_bigram_weight": ${history_proto_bigram_weight:-1.0}
  }
}
JSON

echo "Saved history-sequence eval history -> $history_outfile"
