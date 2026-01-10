#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/deployments/docker/docker-compose.yml}"
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-${CAD_ML_API_PORT:-8000}}"
API_URL="${API_URL:-http://${API_HOST}:${API_PORT}}"
METRICS_URL="${METRICS_URL:-${API_URL}/metrics/}"
API_KEY="${API_KEY:-test}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PROJECT_ROOT/artifacts/docker-staging}"
REPORT_PATH="${REPORT_PATH:-}"
SKIP_BUILD="${SKIP_BUILD:-}"
INSTALL_L3_DEPS="${INSTALL_L3_DEPS:-}"

mkdir -p "$ARTIFACT_DIR"
echo "Artifacts dir: $ARTIFACT_DIR"

COMPOSE_STARTED=0

capture_logs() {
    if [ "$COMPOSE_STARTED" -eq 1 ]; then
        docker compose -f "$COMPOSE_FILE" logs --no-color > "$ARTIFACT_DIR/compose.log" 2>&1 || true
    fi
}

cleanup() {
    if [ "$COMPOSE_STARTED" -eq 1 ]; then
        docker compose -f "$COMPOSE_FILE" down --volumes || true
    fi
}

trap 'capture_logs; cleanup' EXIT

wait_for_url() {
    local url="$1"
    local attempts="${2:-30}"
    local delay="${3:-5}"
    local count=1

    while [ "$count" -le "$attempts" ]; do
        if curl -fsS "$url" > /dev/null; then
            return 0
        fi
        sleep "$delay"
        count=$((count + 1))
    done

    echo "Timed out waiting for $url" >&2
    return 1
}

fetch_metrics_with_retries() {
    local attempts="${1:-5}"
    local delay="${2:-3}"
    local count=1

    while [ "$count" -le "$attempts" ]; do
        if curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused "$METRICS_URL" \
            -o "$ARTIFACT_DIR/metrics.txt"; then
            if grep -q "feature_cache_tuning_requests_total" "$ARTIFACT_DIR/metrics.txt" && \
                grep -q "feature_cache_tuning_recommended_capacity" "$ARTIFACT_DIR/metrics.txt" && \
                grep -q "feature_cache_tuning_recommended_ttl_seconds" "$ARTIFACT_DIR/metrics.txt"; then
                return 0
            fi
        fi
        sleep "$delay"
        count=$((count + 1))
    done

    echo "Cache tuning metrics not found after $attempts attempts" >&2
    if [ -f "$ARTIFACT_DIR/metrics.txt" ]; then
        echo "Metrics snapshot (first 50 lines):" >&2
        head -n 50 "$ARTIFACT_DIR/metrics.txt" >&2 || true
        echo "Cache tuning metrics grep:" >&2
        grep -n "cache_tuning" "$ARTIFACT_DIR/metrics.txt" >&2 || true
    fi
    return 1
}

cd "$PROJECT_ROOT"

echo "Starting docker compose from $COMPOSE_FILE"
if [ -n "$INSTALL_L3_DEPS" ]; then
    docker compose -f "$COMPOSE_FILE" build --build-arg "INSTALL_L3_DEPS=$INSTALL_L3_DEPS"
    SKIP_BUILD=1
fi

compose_args=(-f "$COMPOSE_FILE" up -d)
if [ "$SKIP_BUILD" != "1" ]; then
    compose_args+=(--build)
fi
docker compose "${compose_args[@]}"
COMPOSE_STARTED=1

echo "Waiting for health endpoint at $API_URL/health"
wait_for_url "$API_URL/health" 36 5

curl -fsS "$API_URL/health" -o "$ARTIFACT_DIR/health.json"
python3 - <<'PY' "$ARTIFACT_DIR/health.json"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

metrics_enabled = payload.get("runtime", {}).get("metrics_enabled")
if metrics_enabled is not True:
    raise SystemExit("metrics_enabled is false in /health payload")
PY

echo "Waiting for readiness endpoint at $API_URL/ready"
wait_for_url "$API_URL/ready" 24 5

echo "Posting cache tuning payload"
curl -fsS --retry 3 --retry-delay 2 --retry-connrefused -X POST "$API_URL/api/v1/features/cache/tuning" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"hit_rate":0.32,"capacity":1200,"ttl":3600,"window_hours":6}' \
    -o "$ARTIFACT_DIR/cache_tuning.json"

python3 - <<'PY' "$ARTIFACT_DIR/cache_tuning.json"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

required = ["recommended_capacity", "recommended_ttl", "confidence"]
missing = [name for name in required if name not in payload]
if missing:
    raise SystemExit(f"cache tuning response missing fields: {missing}")
PY

echo "Fetching metrics from $METRICS_URL"
wait_for_url "$METRICS_URL" 24 5
fetch_metrics_with_retries 5 3

echo "Docker staging smoke check completed"

if [ -n "$REPORT_PATH" ]; then
    cat <<EOF_REPORT > "$REPORT_PATH"
# DEV_GITHUB_DOCKER_STAGING_WORKFLOW_VALIDATION

## Scope
Docker-compose staging smoke run for API health, cache tuning endpoint, and metrics exposure.

## Commands
- docker compose -f $COMPOSE_FILE up -d --build
- curl $API_URL/health
- curl -X POST $API_URL/api/v1/features/cache/tuning
- curl $METRICS_URL

## Results
- /health returned metrics_enabled=true.
- Cache tuning endpoint returned recommendation payload.
- Metrics endpoint returned cache tuning metrics:
  - feature_cache_tuning_requests_total
  - feature_cache_tuning_recommended_capacity
  - feature_cache_tuning_recommended_ttl_seconds

## Artifacts
- $ARTIFACT_DIR/health.json
- $ARTIFACT_DIR/cache_tuning.json
- $ARTIFACT_DIR/metrics.txt
- $ARTIFACT_DIR/compose.log
EOF_REPORT
fi
