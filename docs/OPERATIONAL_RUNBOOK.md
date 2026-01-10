# CAD ML Platform - Operational Runbook

## Table of Contents
1. [Service Overview](#service-overview)
2. [Quick Start](#quick-start)
3. [Common Operations](#common-operations)
4. [Monitoring & Health Checks](#monitoring--health-checks)
5. [Incident Response](#incident-response)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Maintenance Procedures](#maintenance-procedures)
9. [Emergency Procedures](#emergency-procedures)
10. [Contact Information](#contact-information)

---

## Service Overview

The CAD ML Platform is a microservice for intelligent CAD drawing analysis, providing OCR extraction and vision understanding capabilities.

### Key Components
- **API Service**: FastAPI application (port 8000)
- **OCR Providers**: PaddleOCR, DeepSeek-HF
- **Vision Provider**: DeepSeek-VL (stub implementation)
- **Metrics**: Prometheus endpoint (port 9090 optional)
- **Cache**: Redis (optional)

### Service Dependencies
```yaml
Internal:
  - Python 3.11+
  - FastAPI framework
  - ML models (loaded on demand)

External (optional):
  - Redis (caching)
  - Prometheus (metrics)
  - Grafana (visualization)
```

---

## Quick Start

### 1. Start Service
```bash
# Development mode
make dev

# Production mode
make serve

# With specific configuration
REDIS_URL=redis://localhost:6379 \
OCR_DEFAULT_PROVIDER=paddle \
make serve
```

### 2. Verify Health
```bash
# Check service health
curl http://localhost:8000/health | jq

# Check readiness
curl http://localhost:8000/ready

# View configuration
curl http://localhost:8000/health | jq '.config'
```

### 3. Monitor Metrics
```bash
# View Prometheus metrics
curl http://localhost:8000/metrics | grep -E "^(ocr|vision)_"

# Export metrics (separate process)
make metrics-export
```

---

## Common Operations

### Starting the Service

#### Development Environment
```bash
# Install dependencies
make install

# Run with auto-reload
make dev

# Run with debug logging
LOG_LEVEL=DEBUG make dev
```

#### Production Environment
```bash
# Pre-flight checks
make test
make security-audit

# Start service
make serve

# Start with custom settings
VISION_MAX_BASE64_BYTES=5242880 \
OCR_TIMEOUT_MS=60000 \
make serve
```

### Stopping the Service

```bash
# Graceful shutdown (SIGTERM)
kill -TERM $(pgrep -f "uvicorn src.main:app")

# Force shutdown (SIGKILL) - use only if graceful fails
kill -KILL $(pgrep -f "uvicorn src.main:app")

# Docker environment
docker-compose down
```

### Configuration Changes

#### Runtime Configuration (via Environment)
```bash
# OCR Configuration
export OCR_DEFAULT_PROVIDER=paddle     # Options: auto, paddle, deepseek_hf
export OCR_TIMEOUT_MS=30000           # OCR processing timeout
export OCR_CONFIDENCE_FALLBACK=0.85   # Default confidence threshold

# Vision Configuration
export VISION_MAX_BASE64_BYTES=1048576  # Max base64 image size (1MB default)

# Redis Configuration
export REDIS_URL=redis://localhost:6379
export REDIS_TTL=3600

# Monitoring
export METRICS_ENABLED=true
export ERROR_EMA_ALPHA=0.2

# Apply and restart
make serve
```

#### Verify Configuration
```bash
# Check active configuration
curl http://localhost:8000/health | jq '.config'

# Validate configuration consistency
python3 scripts/verify_environment.py
```

---

## Monitoring & Health Checks

### Health Endpoints

#### /health - Comprehensive Health Check
```bash
curl http://localhost:8000/health | jq
```

Response includes:
- Service status
- Runtime information (Python version, metrics status)
- Error rate EMA for OCR/Vision
- Complete configuration visibility
- Service dependencies status

#### /ready - Readiness Check
```bash
curl http://localhost:8000/ready
```

Returns 200 if service is ready to handle requests.

#### /metrics - Prometheus Metrics
```bash
curl http://localhost:8000/metrics | head -50
```

### Key Metrics to Monitor

#### Request Metrics
```prometheus
# Total requests by provider and status
ocr_requests_total{provider="paddle",status="success"}
vision_requests_total{provider="deepseek_stub",status="error"}

# Request latency
ocr_processing_duration_seconds_bucket
vision_processing_duration_seconds_bucket
```

#### Error Metrics
```prometheus
# Errors by type
ocr_errors_total{provider="paddle",code="TIMEOUT",stage="infer"}
vision_errors_total{provider="deepseek_stub",code="input_error"}

# Input rejection reasons
ocr_input_rejected_total{reason="file_too_large"}
vision_input_rejected_total{reason="base64_too_large"}

# Error rate EMA (0-1)
error_rate_ema{service="ocr"}
error_rate_ema{service="vision"}
```

#### Performance Metrics
```prometheus
# Stage-wise latency
ocr_stage_duration_seconds{provider="paddle",stage="preprocess"}
ocr_stage_duration_seconds{provider="paddle",stage="infer"}
ocr_stage_duration_seconds{provider="paddle",stage="parse"}

# Confidence scores
ocr_confidence_distribution_bucket
ocr_item_confidence_distribution_bucket

# Circuit breaker status
circuit_breaker_state{circuit="paddle"}  # 0=closed, 1=open, 2=half-open
```

### Alerting Rules

Create these alerts in Prometheus/AlertManager:

```yaml
groups:
  - name: cad_ml_platform
    rules:
      - alert: HighErrorRate
        expr: error_rate_ema{service="ocr"} > 0.1
        for: 5m
        annotations:
          summary: "High OCR error rate: {{ $value }}"

      - alert: ProviderTimeout
        expr: rate(ocr_errors_total{code="PROVIDER_TIMEOUT"}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "Provider {{ $labels.provider }} experiencing timeouts"

      - alert: CircuitOpen
        expr: circuit_breaker_state > 0
        for: 1m
        annotations:
          summary: "Circuit breaker open for {{ $labels.circuit }}"

      - alert: HighMemoryUsage
        expr: rate(ocr_errors_total{code="RESOURCE_EXHAUSTED"}[5m]) > 0
        annotations:
          summary: "Memory exhaustion errors detected"
```

---

## Incident Response

### Error Classification & Response

| Error Code | Severity | Response Time | Action |
|------------|----------|---------------|---------|
| RESOURCE_EXHAUSTED | P0 | Immediate | Scale resources or restart |
| PROVIDER_TIMEOUT | P1 | 15 min | Check provider health |
| RATE_LIMIT | P2 | 1 hour | Review rate limits |
| PARSE_FAILED | P3 | Next day | Review parsing logic |

### Common Incidents

#### 1. Service Unresponsive
```bash
# Check if process is running
ps aux | grep uvicorn

# Check port binding
lsof -i :8000

# Check logs
tail -f logs/app.log

# Restart service
make restart

# If still unresponsive, check resources
free -h
df -h
```

#### 2. High Error Rate
```bash
# Check error metrics
curl http://localhost:8000/metrics | grep error_rate_ema

# Check specific errors
curl http://localhost:8000/metrics | grep ocr_errors_total

# Review recent logs
grep ERROR logs/app.log | tail -20

# Identify problematic provider
curl http://localhost:8000/metrics | grep 'ocr_errors_total{provider'

# Disable problematic provider
OCR_DEFAULT_PROVIDER=paddle make restart
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h
ps aux | grep python | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Check for memory errors
grep RESOURCE_EXHAUSTED logs/app.log

# Clear caches if using Redis
redis-cli FLUSHDB

# Restart with memory limits
ulimit -v 2097152  # 2GB limit
make serve
```

#### 4. Slow Response Times
```bash
# Check stage latencies
curl http://localhost:8000/metrics | grep stage_duration

# Identify slow stages
curl http://localhost:8000/metrics | grep stage_duration | sort -t' ' -k2 -rn | head

# Check provider timeouts
grep TIMEOUT logs/app.log | tail -10

# Increase timeouts if needed
OCR_TIMEOUT_MS=60000 make restart
```

---

## Performance Tuning

### OCR Provider Optimization

```bash
# Use faster provider for development
OCR_DEFAULT_PROVIDER=paddle make serve

# Enable preprocessing for better accuracy
# (Already enabled by default in PaddleOcrProvider)

# Adjust confidence thresholds
OCR_CONFIDENCE_FALLBACK=0.90 make serve
```

### Caching Configuration

```bash
# Enable Redis caching
REDIS_URL=redis://localhost:6379 \
REDIS_TTL=7200 \
make serve

# Monitor cache hit rate
redis-cli INFO stats | grep keyspace_hits
```

### Resource Limits

```bash
# Set memory limits
ulimit -v 4194304  # 4GB

# Set file descriptor limits
ulimit -n 4096

# CPU affinity (bind to specific cores)
taskset -c 0-3 make serve
```

### Connection Pool Tuning

```python
# In production, adjust httpx settings
# src/core/vision/manager.py
async with httpx.AsyncClient(
    timeout=10.0,  # Increase for slow networks
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10
    )
) as client:
```

---

## Troubleshooting Guide

### Diagnosis Tools

#### 1. Health Check Deep Dive
```bash
# Full health with configuration
curl http://localhost:8000/health | jq

# Check specific subsystem
curl http://localhost:8000/health | jq '.services.ml'
```

#### 2. Metrics Analysis
```bash
# Export metrics to file
curl http://localhost:8000/metrics > metrics.txt

# Analyze error patterns
grep -E "ocr_errors_total|vision_errors_total" metrics.txt | \
  awk -F'[{}]' '{print $2}' | sort | uniq -c

# Check confidence distribution
grep confidence_distribution metrics.txt
```

#### 3. Log Analysis
```bash
# Error frequency
grep ERROR logs/app.log | awk '{print $5}' | sort | uniq -c

# Provider issues
grep "provider=" logs/app.log | grep ERROR

# Timing analysis
grep "latency_ms" logs/app.log | \
  awk -F'latency_ms=' '{print $2}' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count}'
```

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Provider not loading | "Provider 'X' not available" | Check dependencies, reinstall packages |
| Base64 decode errors | "Invalid base64" errors | Validate input encoding, check padding |
| Timeout errors | PROVIDER_TIMEOUT in metrics | Increase OCR_TIMEOUT_MS, check model loading |
| Memory errors | RESOURCE_EXHAUSTED | Reduce batch size, increase memory limits |
| Circuit breaker open | Repeated failures | Check provider health, reset circuit manually |
| Cache miss | Low performance | Verify Redis connection, check TTL settings |
| High latency | Slow responses | Profile stages, optimize slowest stage |

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG make dev

# Enable request tracing
TRACE_REQUESTS=true make dev

# Profile specific request
curl -H "X-Trace-Id: debug-123" \
  http://localhost:8000/api/v1/ocr/extract \
  -F "file=@test.png"

# Check trace in logs
grep "debug-123" logs/app.log
```

---

## Maintenance Procedures

### Daily Tasks

```bash
# 1. Check service health
curl http://localhost:8000/health

# 2. Review error rates
curl http://localhost:8000/metrics | grep error_rate_ema

# 3. Check disk usage
df -h /var/log
```

### Weekly Tasks

```bash
# 1. Run evaluation retention
make eval-retention

# 2. Review security
make security-quick

# 3. Clean old logs
find logs/ -mtime +7 -delete

# 4. Update dependencies check
make check-updates
```

### Monthly Tasks

```bash
# 1. Full security audit
make security-audit

# 2. Review badge thresholds
make badges

# 3. Performance baseline
make eval-baseline

# 4. Dependency updates
make update-deps
```

### Quarterly Tasks

```bash
# 1. Baseline snapshot
python3 scripts/snapshot_baseline.py

# 2. Capacity planning review
make eval-phase6

# 3. Documentation update
make docs

# 4. Disaster recovery test
# (Follow emergency procedures)
```

---

## Emergency Procedures

### Service Recovery

#### 1. Immediate Response
```bash
# Check if service is running
systemctl status cad-ml-platform

# Quick restart
systemctl restart cad-ml-platform

# Check logs for errors
journalctl -u cad-ml-platform -n 100
```

#### 2. Rollback Procedure
```bash
# Identify last working version
git log --oneline -10

# Rollback to known good version
git checkout <commit-hash>

# Rebuild and restart
make clean install serve
```

#### 3. Data Recovery
```bash
# Backup current state
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/

# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz

# Verify data integrity
python3 scripts/check_integrity.py
```

### Disaster Recovery

#### Complete Service Rebuild
```bash
# 1. Clone repository
git clone https://github.com/org/cad-ml-platform.git
cd cad-ml-platform

# 2. Install dependencies
make install

# 3. Restore configuration
cp /backup/config/.env .env

# 4. Restore data
cp -r /backup/data/* data/

# 5. Verify setup
make test

# 6. Start service
make serve
```

#### Database Recovery (if using Redis)
```bash
# Backup Redis
redis-cli BGSAVE

# Restore Redis
service redis stop
cp /backup/dump.rdb /var/lib/redis/
service redis start
```

---

## Contact Information

### Escalation Path

| Level | Role | Contact | Response Time |
|-------|------|---------|---------------|
| L1 | On-call Engineer | PagerDuty | 15 min |
| L2 | Platform Team | #platform-team | 1 hour |
| L3 | ML Team | #ml-team | 4 hours |
| L4 | Architecture | #architecture | Next day |

### Key Contacts

- **Service Owner**: ML Platform Team
- **Slack Channel**: #cad-ml-platform
- **Email**: ml-platform@company.com
- **Wiki**: https://wiki.company.com/cad-ml-platform
- **Repository**: https://github.com/org/cad-ml-platform
- **Monitoring Dashboard**: https://grafana.company.com/cad-ml

### External Dependencies

| Service | Contact | SLA |
|---------|---------|-----|
| Redis | Infrastructure Team | 99.9% |
| Prometheus | Monitoring Team | 99.5% |
| Model Storage | ML Infrastructure | 99.9% |

---

## Appendix

### Environment Variables Reference

```bash
# Core Service
PORT=8000
WORKERS=4
LOG_LEVEL=INFO

# OCR Configuration
OCR_DEFAULT_PROVIDER=auto
OCR_TIMEOUT_MS=30000
OCR_CONFIDENCE_FALLBACK=0.85
OCR_CACHE_TTL=3600

# Vision Configuration
VISION_MAX_BASE64_BYTES=1048576
VISION_TIMEOUT_MS=5000

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_TTL=3600
REDIS_MAX_CONNECTIONS=10

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
ERROR_EMA_ALPHA=0.2

# Circuit Breaker
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_RECOVERY_TIMEOUT=60
CIRCUIT_EXPECTED_EXCEPTION=TimeoutError

# Rate Limiting
RATE_LIMIT_QPS=10
RATE_LIMIT_BURST=10
```

### Useful Commands

```bash
# Service Management
make serve          # Start production server
make dev           # Start development server
make restart       # Restart service
make stop          # Stop service

# Testing & Validation
make test          # Run all tests
make test-fast     # Run fast tests only
make lint          # Run linters
make typecheck     # Run type checking
# Faiss perf tests (optional)
RUN_FAISS_PERF_TESTS=1 pytest tests/perf/test_vector_search_latency.py -v
REQUIRE_FAISS_PERF=1 RUN_FAISS_PERF_TESTS=1 pytest tests/perf/test_vector_search_latency.py -v
# Note: some environments segfault on faiss import with PYTHONWARNINGS=error::DeprecationWarning.
# The perf test runs faiss in a subprocess and filters swig DeprecationWarning to isolate this.

# Monitoring
make metrics-export  # Export metrics to Prometheus
make health-check   # Check service health
make eval-phase6    # Run full evaluation

# Maintenance
make clean         # Clean build artifacts
make install       # Install dependencies
make update-deps   # Update dependencies
make security-audit # Run security scan

# Documentation
make docs          # Generate documentation
make badges        # Generate status badges
```

### Performance Benchmarks

| Operation | Expected Latency | Timeout |
|-----------|-----------------|---------|
| OCR Extraction | < 2s | 30s |
| Vision Analysis | < 500ms | 5s |
| Health Check | < 100ms | 1s |
| Metrics Export | < 200ms | 2s |

### Capacity Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Max Image Size | 50MB (URL), 1MB (base64) | Configurable |
| Max Concurrent Requests | 100 | Per worker |
| Max Memory per Worker | 2GB | Monitor for RESOURCE_EXHAUSTED |
| Cache Size | 1GB | Redis memory limit |
| Log Retention | 7 days | Rotate daily |

---

*Last Updated: 2025-11-20*
*Version: 1.0.0*
