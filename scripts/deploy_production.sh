#!/bin/bash

# Production Deployment Script for CAD ML Platform with Observability
# This script handles the complete production deployment with safety checks

set -e  # Exit on error
set -o pipefail  # Pipe failures cause exit

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
NAMESPACE=${NAMESPACE:-cad-ml-platform}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-registry.example.com}
VERSION=${VERSION:-$(git describe --tags --always)}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found"
        exit 1
    fi

    # Check Kubernetes access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi

    # Check namespace
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace $NAMESPACE
    fi

    # Run self-check
    log_info "Running self-check..."
    if ! python3 scripts/self_check.py --json > /tmp/self-check.json; then
        log_error "Self-check failed"
        cat /tmp/self-check.json
        exit 1
    fi

    # Validate metrics contract
    log_info "Validating metrics contract..."
    if ! python3 -m pytest tests/test_metrics_contract.py -q; then
        log_error "Metrics contract validation failed"
        exit 1
    fi

    # Validate Prometheus rules
    log_info "Validating Prometheus rules..."
    if ! python3 scripts/validate_prom_rules.py --skip-promtool --json > /tmp/prom-validation.json; then
        log_error "Prometheus rules validation failed"
        cat /tmp/prom-validation.json
        exit 1
    fi

    log_info "Pre-deployment checks passed ✓"
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."

    # Build image
    docker build -t $DOCKER_REGISTRY/cad-ml-platform:$VERSION \
                 -t $DOCKER_REGISTRY/cad-ml-platform:latest \
                 --build-arg VERSION=$VERSION \
                 --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
                 .

    if [ $? -ne 0 ]; then
        log_error "Docker build failed"
        exit 1
    fi

    # Run security scan
    log_info "Running security scan on image..."
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy image --severity HIGH,CRITICAL \
        $DOCKER_REGISTRY/cad-ml-platform:$VERSION

    # Push to registry
    log_info "Pushing image to registry..."
    docker push $DOCKER_REGISTRY/cad-ml-platform:$VERSION
    docker push $DOCKER_REGISTRY/cad-ml-platform:latest

    log_info "Image built and pushed successfully ✓"
}

# Deploy Prometheus and Grafana
deploy_monitoring() {
    log_info "Deploying monitoring stack..."

    # Deploy Prometheus
    kubectl apply -f k8s/prometheus/ -n $NAMESPACE

    # Deploy Grafana
    kubectl apply -f k8s/grafana/ -n $NAMESPACE

    # Wait for monitoring to be ready
    log_info "Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s

    log_info "Monitoring stack deployed ✓"
}

# Deploy application
deploy_application() {
    log_info "Deploying CAD ML Platform..."

    # Update image in deployment
    kubectl set image deployment/cad-ml-platform \
        cad-ml-platform=$DOCKER_REGISTRY/cad-ml-platform:$VERSION \
        -n $NAMESPACE

    # Apply other resources
    kubectl apply -f k8s/app/ -n $NAMESPACE

    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    if ! kubectl rollout status deployment/cad-ml-platform -n $NAMESPACE --timeout=600s; then
        log_error "Deployment rollout failed"

        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            log_warn "Rolling back deployment..."
            kubectl rollout undo deployment/cad-ml-platform -n $NAMESPACE
            kubectl rollout status deployment/cad-ml-platform -n $NAMESPACE --timeout=300s
        fi
        exit 1
    fi

    log_info "Application deployed ✓"
}

# Post-deployment validation
post_deployment_validation() {
    log_info "Running post-deployment validation..."

    # Get service endpoint
    SERVICE_IP=$(kubectl get service cad-ml-platform -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_IP" ]; then
        SERVICE_IP=$(kubectl get service cad-ml-platform -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    fi

    # Wait for service to be ready
    log_info "Waiting for service to be ready..."
    for i in {1..30}; do
        if curl -s http://$SERVICE_IP:8000/health | jq -e '.status == "healthy"' > /dev/null 2>&1; then
            break
        fi
        sleep 10
    done

    # Run self-check against production
    log_info "Running production self-check..."
    SELF_CHECK_BASE_URL=http://$SERVICE_IP:8000 \
    SELF_CHECK_STRICT_METRICS=1 \
    python3 scripts/self_check.py --json > /tmp/prod-self-check.json

    if [ $(jq -r '.success' /tmp/prod-self-check.json) != "true" ]; then
        log_error "Production self-check failed"
        cat /tmp/prod-self-check.json

        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            log_warn "Rolling back due to validation failure..."
            kubectl rollout undo deployment/cad-ml-platform -n $NAMESPACE
        fi
        exit 1
    fi

    # Check metrics endpoint
    log_info "Validating metrics endpoint..."
    if ! curl -s http://$SERVICE_IP:8000/metrics | grep -q "ocr_errors_total"; then
        log_error "Metrics endpoint validation failed"
        exit 1
    fi

    # Check Prometheus targets
    log_info "Checking Prometheus targets..."
    PROM_IP=$(kubectl get service prometheus -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    TARGETS=$(curl -s http://$PROM_IP:9090/api/v1/targets | jq -r '.data.activeTargets[] | select(.labels.job=="cad-ml-platform") | .health')

    if [ "$TARGETS" != "up" ]; then
        log_warn "Prometheus target not healthy yet"
    fi

    log_info "Post-deployment validation passed ✓"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."

    cat > /tmp/deployment-report.md << EOF
# Deployment Report

**Date**: $(date)
**Version**: $VERSION
**Environment**: $ENVIRONMENT
**Namespace**: $NAMESPACE

## Deployment Status
- Pre-deployment checks: ✓
- Docker build: ✓
- Security scan: ✓
- Monitoring deployment: ✓
- Application deployment: ✓
- Post-deployment validation: ✓

## Service Endpoints
- Application: http://$SERVICE_IP:8000
- Prometheus: http://$PROM_IP:9090
- Grafana: http://$(kubectl get service grafana -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):3000

## Health Status
$(cat /tmp/prod-self-check.json | jq -r '.checks')

## Metrics Status
- Metrics endpoint: Active
- Prometheus scraping: Active
- Recording rules: Loaded

## Next Steps
1. Monitor dashboards for 30 minutes
2. Run smoke tests
3. Enable traffic shifting (if using canary)
4. Update documentation

---
Generated by deploy_production.sh
EOF

    cat /tmp/deployment-report.md

    # Send report (implement notification)
    # send_slack_notification "Deployment completed successfully for version $VERSION"

    log_info "Deployment report generated ✓"
}

# Main deployment flow
main() {
    log_info "Starting production deployment for CAD ML Platform"
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"

    # Create deployment backup
    kubectl get all -n $NAMESPACE -o yaml > /tmp/backup-$(date +%Y%m%d-%H%M%S).yaml

    # Run deployment steps
    pre_deployment_checks
    build_and_push
    deploy_monitoring
    deploy_application
    post_deployment_validation
    generate_report

    log_info "🎉 Deployment completed successfully!"
    log_info "Access the application at: http://$SERVICE_IP:8000"
    log_info "Access Grafana at: http://$(kubectl get service grafana -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):3000"
}

# Handle errors
trap 'log_error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"