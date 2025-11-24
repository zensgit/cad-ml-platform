#!/usr/bin/env bash
set -euo pipefail

# Validate Prometheus rules using promtool (local or Docker fallback)
# Usage: scripts/validate_prometheus.sh [rule_file ...]

# Default rule files (alert + recording). Add new drift baseline stale check.
RULES=(
  "config/prometheus/alerting_rules.yml"
  "prometheus/rules/cad_ml_recording_rules.yml"
  "docs/prometheus/recording_rules.yml"
)

if [[ "$#" -gt 0 ]]; then
  RULES=("$@")
fi

have_promtool=0
if command -v promtool >/dev/null 2>&1; then
  have_promtool=1
fi

function validate_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[skip] $f (not found)"
    return 0
  fi
  echo "[check] $f"
  if [[ $have_promtool -eq 1 ]]; then
    promtool check rules "$f"
  else
    docker run --rm -v "$(pwd)":/workspace:ro prom/prometheus:latest promtool check rules "/workspace/$f"
  fi
}

for f in "${RULES[@]}"; do
  validate_file "$f"
done

echo "All rule files validated."
