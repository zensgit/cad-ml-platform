#!/usr/bin/env bash
set -euo pipefail

# Save OCR golden evaluation metrics to JSON under reports/eval_history/
# Also captures git metadata for trend analysis.

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
REPORT_DIR="$ROOT_DIR/reports/eval_history"
SCRIPT="$ROOT_DIR/tests/ocr/run_golden_evaluation.py"

mkdir -p "$REPORT_DIR"

echo "Running OCR golden evaluation..."
python3 "$SCRIPT" | tee /tmp/eval_out.txt

timestamp=$(date +"%Y%m%d_%H%M%S")
branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
safe_branch=$(echo "$branch" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+|-+$//g')
safe_branch=${safe_branch:-unknown}
commit=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Parse metrics from stdout
dimension_recall=$(grep -Eo 'dimension_recall=[0-9.]+"?' /tmp/eval_out.txt | cut -d= -f2)
brier_score=$(grep -Eo 'brier_score=[0-9.]+' /tmp/eval_out.txt | cut -d= -f2)
edge_f1=$(grep -Eo 'edge_f1=[0-9.]+' /tmp/eval_out.txt | cut -d= -f2)

outfile="$REPORT_DIR/${timestamp}_${safe_branch}_${commit}.json"

# Get system information for run context
runner="local"
[[ -n "${CI:-}" ]] && runner="ci"
machine_name=$(hostname 2>/dev/null || echo "unknown")
os_info=$(uname -s 2>/dev/null || echo "unknown")
os_version=$(uname -r 2>/dev/null || echo "unknown")
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
start_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

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
