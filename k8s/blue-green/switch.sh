#!/bin/bash
# Blue-Green Deployment Script for CAD ML Platform
# Usage: ./switch.sh [blue|green]

set -euo pipefail

NAMESPACE="${NAMESPACE:-cad-ml-platform}"
TARGET_SLOT="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_current_slot() {
    kubectl get svc cad-ml-api -n "$NAMESPACE" -o jsonpath='{.spec.selector.slot}' 2>/dev/null || echo "unknown"
}

get_deployment_status() {
    local slot=$1
    kubectl get deployment "cad-ml-api-$slot" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}/{.status.replicas}' 2>/dev/null || echo "0/0"
}

wait_for_deployment() {
    local slot=$1
    local timeout=${2:-300}
    local start=$(date +%s)

    log_info "Waiting for $slot deployment to be ready (timeout: ${timeout}s)..."

    while true; do
        local status=$(get_deployment_status "$slot")
        local ready=$(echo "$status" | cut -d'/' -f1)
        local total=$(echo "$status" | cut -d'/' -f2)

        if [[ "$ready" == "$total" && "$total" != "0" ]]; then
            log_info "$slot deployment is ready ($status)"
            return 0
        fi

        local elapsed=$(($(date +%s) - start))
        if [[ $elapsed -ge $timeout ]]; then
            log_error "Timeout waiting for $slot deployment"
            return 1
        fi

        echo -ne "\r  Status: $status (${elapsed}s elapsed)"
        sleep 5
    done
}

health_check() {
    local slot=$1
    local service="cad-ml-api-$slot"

    log_info "Running health check on $slot..."

    # Port-forward and check health
    kubectl port-forward "svc/$service" 8080:80 -n "$NAMESPACE" &
    local pf_pid=$!
    sleep 3

    local health_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo "000")
    kill $pf_pid 2>/dev/null || true

    if [[ "$health_status" == "200" ]]; then
        log_info "Health check passed (HTTP $health_status)"
        return 0
    else
        log_error "Health check failed (HTTP $health_status)"
        return 1
    fi
}

switch_traffic() {
    local target_slot=$1

    log_info "Switching traffic to $target_slot..."

    kubectl patch svc cad-ml-api -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"app\":\"cad-ml-api\",\"slot\":\"$target_slot\"}}}"

    log_info "Traffic switched to $target_slot"
}

scale_deployment() {
    local slot=$1
    local replicas=$2

    log_info "Scaling $slot to $replicas replicas..."
    kubectl scale deployment "cad-ml-api-$slot" -n "$NAMESPACE" --replicas="$replicas"
}

# Main execution
main() {
    log_info "Blue-Green Deployment Manager"
    log_info "=============================="

    local current_slot=$(get_current_slot)
    log_info "Current active slot: $current_slot"

    if [[ -z "$TARGET_SLOT" ]]; then
        # Auto-detect target slot
        if [[ "$current_slot" == "blue" ]]; then
            TARGET_SLOT="green"
        else
            TARGET_SLOT="blue"
        fi
        log_info "Auto-selected target slot: $TARGET_SLOT"
    fi

    if [[ "$TARGET_SLOT" != "blue" && "$TARGET_SLOT" != "green" ]]; then
        log_error "Invalid slot: $TARGET_SLOT (must be 'blue' or 'green')"
        exit 1
    fi

    if [[ "$TARGET_SLOT" == "$current_slot" ]]; then
        log_warn "Target slot is already active"
        exit 0
    fi

    # Scale up target deployment
    scale_deployment "$TARGET_SLOT" 3

    # Wait for target to be ready
    if ! wait_for_deployment "$TARGET_SLOT"; then
        log_error "Failed to deploy to $TARGET_SLOT, aborting"
        scale_deployment "$TARGET_SLOT" 0
        exit 1
    fi

    # Health check
    if ! health_check "$TARGET_SLOT"; then
        log_error "Health check failed for $TARGET_SLOT, aborting"
        scale_deployment "$TARGET_SLOT" 0
        exit 1
    fi

    # Switch traffic
    switch_traffic "$TARGET_SLOT"

    # Wait for connections to drain
    log_info "Waiting 30s for connections to drain from $current_slot..."
    sleep 30

    # Scale down old deployment
    scale_deployment "$current_slot" 0

    log_info "=============================="
    log_info "Deployment complete!"
    log_info "Active slot: $TARGET_SLOT"
    log_info "Previous slot ($current_slot) scaled down"
}

# Rollback command
if [[ "${1:-}" == "rollback" ]]; then
    current_slot=$(get_current_slot)
    if [[ "$current_slot" == "blue" ]]; then
        TARGET_SLOT="green"
    else
        TARGET_SLOT="blue"
    fi
    log_warn "Rolling back to $TARGET_SLOT"
    main
    exit 0
fi

main
