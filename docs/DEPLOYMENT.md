# CAD ML Platform - Deployment Guide

## Prerequisites

- Docker 20.10+
- Kubernetes 1.25+ (for K8s deployment)
- Helm 3.10+ (for Helm deployment)
- Redis 6.0+
- 4GB RAM minimum, 8GB recommended

## Quick Start with Docker

### 1. Build the Image

```bash
# Production image
docker build -t cad-ml-platform:latest --target production .

# Development image (includes dev dependencies)
docker build -t cad-ml-platform:dev --target development .

# GPU image (CUDA 11.8)
docker build -t cad-ml-platform:gpu --target gpu-base .
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    image: cad-ml-platform:latest
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
      - APP_ENV=production
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: cad-ml-platform:latest
    command: arq src.core.dedupcad_2d_worker.WorkerSettings
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

## Kubernetes Deployment

### 1. Using Helm Chart

```bash
# Add Helm repository (if external)
# helm repo add cad-ml https://charts.example.com

# Install with default values
helm install cad-ml charts/cad-ml-platform -n cad-ml --create-namespace

# Install with custom values
helm install cad-ml charts/cad-ml-platform \
  -n cad-ml \
  --create-namespace \
  -f values-production.yaml
```

### 2. Custom Values

**values-production.yaml:**
```yaml
replicaCount: 3

image:
  repository: ghcr.io/example/cad-ml-platform
  tag: "2.0.0"
  pullPolicy: IfNotPresent

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cad-ml-tls
      hosts:
        - api.example.com

env:
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  REDIS_URL: "redis://redis-master:6379"
  CACHE_L1_ENABLED: "true"
  CACHE_L2_ENABLED: "true"

redis:
  enabled: true
  architecture: replication
  auth:
    enabled: true
    password: "your-redis-password"

metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring
```

### 3. Manual Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace cad-ml

# Apply configurations
kubectl apply -f deploy/k8s/ -n cad-ml

# Check deployment status
kubectl get pods -n cad-ml
kubectl get svc -n cad-ml
kubectl get hpa -n cad-ml
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/production) | `production` |
| `APP_PORT` | Application port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | Log format (json/text) | `json` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `REDIS_ENABLED` | Enable Redis caching | `true` |
| `CACHE_L1_ENABLED` | Enable L1 (in-memory) cache | `true` |
| `CACHE_L1_MAX_SIZE` | L1 cache max entries | `2000` |
| `CACHE_L2_ENABLED` | Enable L2 (Redis) cache | `true` |
| `MODEL_PATH` | Model files directory | `/models` |
| `VECTOR_STORE_BACKEND` | Vector store (faiss/milvus) | `faiss` |

## Health Checks

### Liveness Probe
```
GET /health
```
Returns 200 if the application is running.

### Readiness Probe
```
GET /api/v1/health
```
Returns 200 if all dependencies are healthy.

### Startup Probe
Allows up to 150 seconds for model loading before failing.

## Scaling Guidelines

### Horizontal Scaling

The platform scales horizontally. Key considerations:

1. **API Pods**: Scale based on CPU/memory usage
2. **Worker Pods**: Scale based on queue depth
3. **Redis**: Use Redis Cluster for high availability

```yaml
autoscaling:
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### Vertical Scaling

For ML-heavy workloads:

```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
    nvidia.com/gpu: 1  # For GPU nodes
  limits:
    cpu: 4000m
    memory: 8Gi
    nvidia.com/gpu: 1
```

## Monitoring Setup

### 1. Prometheus

Add ServiceMonitor for metrics scraping:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: cad-ml-platform
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: cad-ml-platform
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

### 2. Grafana Dashboard

Import the dashboard from:
```
deploy/monitoring/grafana/cad-ml-platform-dashboard.json
```

### 3. Alerting

Apply AlertManager rules:
```bash
kubectl apply -f deploy/monitoring/alertmanager/alerts.yaml -n monitoring
```

## Security Hardening

### 1. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cad-ml-platform
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: cad-ml-platform
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - port: 6379
```

### 2. Pod Security

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

containerSecurityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
```

### 3. Secrets Management

Use external secrets operator or sealed secrets:

```bash
# Create secret
kubectl create secret generic cad-ml-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=redis-password=your-redis-password \
  -n cad-ml
```

## Backup and Recovery

### Redis Backup

```bash
# Manual backup
kubectl exec redis-master-0 -n cad-ml -- redis-cli BGSAVE

# Copy backup
kubectl cp cad-ml/redis-master-0:/data/dump.rdb ./backup/redis-dump.rdb
```

### Model Backup

```bash
# Backup models PVC
kubectl get pvc models-pvc -n cad-ml -o yaml > backup/models-pvc.yaml

# Use velero for full backup
velero backup create cad-ml-backup --include-namespaces cad-ml
```

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
   ```bash
   kubectl logs -f <pod-name> -n cad-ml --previous
   kubectl describe pod <pod-name> -n cad-ml
   ```

2. **High Memory Usage**
   - Check model cache size
   - Verify L1 cache configuration
   - Review batch job concurrency

3. **Slow API Responses**
   - Check Redis connectivity
   - Review HPA scaling status
   - Verify model loading status

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/cad-ml-platform LOG_LEVEL=DEBUG -n cad-ml

# Port forward for local debugging
kubectl port-forward svc/cad-ml-platform 8000:8000 -n cad-ml
```

## Rollback

```bash
# Helm rollback
helm rollback cad-ml 1 -n cad-ml

# Kubernetes rollback
kubectl rollout undo deployment/cad-ml-platform -n cad-ml

# Check rollout history
kubectl rollout history deployment/cad-ml-platform -n cad-ml
```
