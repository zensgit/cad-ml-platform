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


inputs = load_json_env("GITHUB_EVENT_INPUTS_JSON")
mapping = {
    "WF_INPUT_BREP_GOLDEN_MANIFEST_ENABLE": "brep_golden_manifest_enable",
    "WF_INPUT_BREP_GOLDEN_MANIFEST_JSON": "brep_golden_manifest_json",
    "WF_INPUT_BREP_GOLDEN_MANIFEST_OUTPUT_JSON": "brep_golden_manifest_output_json",
    "WF_INPUT_BREP_GOLDEN_MANIFEST_MIN_RELEASE_SAMPLES": (
        "brep_golden_manifest_min_release_samples"
    ),
    "WF_INPUT_BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY": (
        "brep_golden_manifest_fail_on_not_release_ready"
    ),
    "WF_INPUT_BREP_GOLDEN_EVAL_ENABLE": "brep_golden_eval_enable",
    "WF_INPUT_BREP_GOLDEN_EVAL_OUTPUT_DIR": "brep_golden_eval_output_dir",
    "WF_INPUT_BREP_GOLDEN_EVAL_STRICT": "brep_golden_eval_strict",
    "WF_INPUT_BREP_GOLDEN_EVAL_ALLOW_DEMO_GEOMETRY": (
        "brep_golden_eval_allow_demo_geometry"
    ),
}

for env_name, input_name in mapping.items():
    value = str(inputs.get(input_name) or "")
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

resolve_setting() {
  local input_value="$1"
  local env_value="$2"
  local default_value="$3"
  if [[ -n "$input_value" ]]; then
    printf '%s' "$input_value"
  elif [[ -n "$env_value" ]]; then
    printf '%s' "$env_value"
  else
    printf '%s' "$default_value"
  fi
}

MANIFEST_ENABLE_VALUE="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_MANIFEST_ENABLE:-}" \
    "${BREP_GOLDEN_MANIFEST_ENABLE:-}" \
    "false"
)"
MANIFEST_JSON="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_MANIFEST_JSON:-}" \
    "${BREP_GOLDEN_MANIFEST_JSON:-}" \
    "config/brep_golden_manifest.example.json"
)"
VALIDATION_JSON="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_MANIFEST_OUTPUT_JSON:-}" \
    "${BREP_GOLDEN_MANIFEST_OUTPUT_JSON:-}" \
    "reports/benchmark/brep_golden_manifest/latest_validation.json"
)"
MIN_RELEASE_SAMPLES="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_MANIFEST_MIN_RELEASE_SAMPLES:-}" \
    "${BREP_GOLDEN_MANIFEST_MIN_RELEASE_SAMPLES:-}" \
    "50"
)"
FAIL_ON_NOT_RELEASE_READY="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY:-}" \
    "${BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY:-}" \
    "false"
)"
EVAL_ENABLE_VALUE="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_EVAL_ENABLE:-}" \
    "${BREP_GOLDEN_EVAL_ENABLE:-}" \
    "false"
)"
EVAL_OUTPUT_DIR="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_EVAL_OUTPUT_DIR:-}" \
    "${BREP_GOLDEN_EVAL_OUTPUT_DIR:-}" \
    "reports/benchmark/brep_step_iges_golden/latest"
)"
EVAL_STRICT_VALUE="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_EVAL_STRICT:-}" \
    "${BREP_GOLDEN_EVAL_STRICT:-}" \
    "true"
)"
ALLOW_DEMO_GEOMETRY="$(
  resolve_setting \
    "${WF_INPUT_BREP_GOLDEN_EVAL_ALLOW_DEMO_GEOMETRY:-}" \
    "${BREP_GOLDEN_EVAL_ALLOW_DEMO_GEOMETRY:-}" \
    "false"
)"

if ! is_true "$MANIFEST_ENABLE_VALUE" \
  && ! is_true "$EVAL_ENABLE_VALUE" \
  && ! is_true "$FAIL_ON_NOT_RELEASE_READY"; then
  write_output enabled false
  write_output eval_enabled false
  echo "BREP_GOLDEN_MANIFEST_ENABLE is not true; skip."
  exit 0
fi

write_output enabled true
write_output manifest_json "$MANIFEST_JSON"
write_output validation_json "$VALIDATION_JSON"
write_output eval_enabled false

mkdir -p "$(dirname "$VALIDATION_JSON")"

VALIDATE_CMD=(
  python3 scripts/validate_brep_golden_manifest.py
  --manifest "$MANIFEST_JSON"
  --output-json "$VALIDATION_JSON"
  --min-release-samples "$MIN_RELEASE_SAMPLES"
)
if is_true "$FAIL_ON_NOT_RELEASE_READY"; then
  VALIDATE_CMD+=(--fail-on-not-release-ready)
fi

VALIDATION_EXIT=0
"${VALIDATE_CMD[@]}" || VALIDATION_EXIT=$?

if [[ -f "$VALIDATION_JSON" ]]; then
  BREP_GOLDEN_MANIFEST_VALIDATION_JSON="$VALIDATION_JSON" python3 - <<'PY'
import json
import os
from pathlib import Path

output = os.environ.get("GITHUB_OUTPUT")
if output:
    payload = json.loads(
        Path(os.environ["BREP_GOLDEN_MANIFEST_VALIDATION_JSON"]).read_text(
            encoding="utf-8"
        )
    )
    ready_for_release = str(bool(payload.get("ready_for_release", False))).lower()
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(f"validation_status={payload.get('status', 'unknown')}\n")
        handle.write(f"ready_for_release={ready_for_release}\n")
        handle.write(f"case_count={payload.get('case_count', 0)}\n")
        handle.write(
            f"release_eligible_count={payload.get('release_eligible_count', 0)}\n"
        )
        handle.write(f"min_release_samples={payload.get('min_release_samples', 0)}\n")
PY
else
  write_output validation_status missing
fi

if [[ "$VALIDATION_EXIT" -ne 0 ]]; then
  echo "B-Rep golden manifest validation failed with exit code $VALIDATION_EXIT."
  exit "$VALIDATION_EXIT"
fi

if is_true "$EVAL_ENABLE_VALUE"; then
  mkdir -p "$EVAL_OUTPUT_DIR"
  EVAL_CMD=(
    python3 scripts/eval_brep_step_dir.py
    --manifest "$MANIFEST_JSON"
    --output-dir "$EVAL_OUTPUT_DIR"
  )
  if is_true "$EVAL_STRICT_VALUE"; then
    EVAL_CMD+=(--strict)
  fi
  if is_true "$ALLOW_DEMO_GEOMETRY"; then
    EVAL_CMD+=(--allow-demo-geometry)
  fi

  "${EVAL_CMD[@]}"

  write_output eval_enabled true
  write_output eval_output_dir "$EVAL_OUTPUT_DIR"
  write_output eval_summary_json "$EVAL_OUTPUT_DIR/summary.json"
  write_output eval_results_csv "$EVAL_OUTPUT_DIR/results.csv"
  write_output eval_graph_qa_json "$EVAL_OUTPUT_DIR/graph_qa.json"
  BREP_GOLDEN_EVAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

output = os.environ.get("GITHUB_OUTPUT")
output_dir = Path(os.environ["BREP_GOLDEN_EVAL_OUTPUT_DIR"])
summary_path = output_dir / "summary.json"
graph_path = output_dir / "graph_qa.json"
if output and summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}
    with open(output, "a", encoding="utf-8") as handle:
        handle.write(f"eval_sample_size={summary.get('sample_size', 0)}\n")
        handle.write(f"eval_parse_success_count={summary.get('parse_success_count', 0)}\n")
        handle.write(f"eval_graph_valid_count={summary.get('graph_valid_count', 0)}\n")
        handle.write(f"eval_graph_qa_status={graph.get('status', 'unknown')}\n")
PY
fi
