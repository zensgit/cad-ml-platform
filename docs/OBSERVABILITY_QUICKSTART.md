# 🚀 Observability Quick Start Guide

## 📋 Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ installed
- 8GB+ RAM available
- Ports available: 8000, 9090, 3000, 6379

## 🎯 5-Minute Setup

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/cad-ml-platform.git
cd cad-ml-platform

# Install Python dependencies
make install
```

### Step 2: Start Observability Stack

```bash
# Start everything with one command
make observability-up

# Or using docker-compose directly
docker-compose -f docker-compose.observability.yml up -d
```

This starts:
- 🚀 **CAD ML Platform** (port 8000)
- 📊 **Prometheus** (port 9090)
- 📈 **Grafana** (port 3000)
- 💾 **Redis** (port 6379)

### Step 3: Verify Installation

```bash
# Check all services are running
make observability-status

# Run self-check
make self-check

# Or strict mode
make self-check-strict
```

### Step 4: Access Dashboards

1. **Application**: http://localhost:8000
   - Health: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics
   - API Docs: http://localhost:8000/docs

2. **Prometheus**: http://localhost:9090
   - Targets: http://localhost:9090/targets
   - Rules: http://localhost:9090/rules
   - Query: http://localhost:9090/graph

3. **Grafana**: http://localhost:3000
   - Login: `admin` / `admin`
   - Dashboard: Pre-imported "CAD ML Platform - Observability"

## 🔧 Common Operations

### Generate Some Metrics

```bash
# Make API calls to generate metrics
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@sample.pdf"

# Trigger an error for testing
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -F "file=@invalid.txt"

# Check metrics
curl http://localhost:8000/metrics | grep ocr_errors_total
```

### Validate Metrics Contract

```bash
# Run metrics validation
make metrics-validate

# Validate Prometheus rules
make prom-validate

# Run full observability test suite
make observability-test
```

### View Logs

```bash
# Stream all logs
make observability-logs

# Or specific service
docker-compose -f docker-compose.observability.yml logs -f app
```

## 📊 Key Metrics to Monitor

### In Prometheus (http://localhost:9090/graph)

```promql
# Error rate
rate(ocr_errors_total[5m])

# Request latency (P95)
histogram_quantile(0.95, rate(ocr_processing_duration_seconds_bucket[5m]))

# Provider health
up{job="cad-ml-platform"}

# Recording rule - error ratio
ocr_error_ratio
```

### In Grafana Dashboard

1. **Platform Error Rates** - Overall system health
2. **Provider Error Breakdown** - By provider and error code
3. **Input Rejection Rates** - Invalid input tracking
4. **Provider Health Score** - 0-100 score per provider
5. **SLO Compliance** - Service level objectives
6. **Error Budget** - Remaining error budget

## 🔍 Troubleshooting

### Service Won't Start

```bash
# Check port conflicts
lsof -i :8000
lsof -i :9090
lsof -i :3000

# Check Docker resources
docker system df
docker system prune -a  # Clean up if needed
```

### Metrics Not Appearing

```bash
# Check app metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart services
make observability-restart
```

### Dashboard Shows No Data

1. Check Prometheus is scraping:
   ```bash
   curl http://localhost:9090/api/v1/query?query=up
   ```

2. Verify recording rules:
   ```bash
   curl http://localhost:9090/api/v1/rules
   ```

3. Re-import dashboard:
   ```bash
   make dashboard-import
   ```

## 🧪 Testing the Stack

### Basic Test Flow

```bash
# 1. Start the stack
make observability-up

# 2. Wait for services
sleep 30

# 3. Run self-check
python scripts/self_check.py --json

# 4. Generate test load
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/ocr/extract \
    -F "file=@test.pdf" &
done
wait

# 5. Check metrics
curl http://localhost:8000/metrics | grep -E "ocr_requests_total|ocr_errors_total"

# 6. View in Grafana
open http://localhost:3000
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Start Observability Stack
  run: make observability-up

- name: Run Self-Check
  run: |
    make self-check-json > result.json
    if [ $(jq -r '.success' result.json) != "true" ]; then
      exit 1
    fi

- name: Validate Metrics
  run: make metrics-validate

- name: Stop Stack
  run: make observability-down
```

## 📚 Learn More

### Essential Commands

| Command | Description |
|---------|-------------|
| `make obs-up` | Start observability stack |
| `make obs-down` | Stop observability stack |
| `make obs-status` | Check service health |
| `make self-check` | Basic health check |
| `make self-check-strict` | Strict metrics validation |
| `make metrics-validate` | Test metrics contract |
| `make prom-validate` | Validate recording rules |
| `make observability-test` | Run full test suite |

### Configuration Files

- `docker-compose.observability.yml` - Stack definition
- `config/prometheus.yml` - Prometheus config
- `docs/prometheus/recording_rules.yml` - Recording rules
- `docs/grafana/observability_dashboard.json` - Dashboard
- `scripts/self_check.py` - Self-check script

### Environment Variables

```bash
# Self-check configuration
export SELF_CHECK_STRICT_METRICS=1
export SELF_CHECK_MIN_OCR_ERRORS=5
export SELF_CHECK_BASE_URL=http://production:8000

# Run with custom settings
python scripts/self_check.py --json
```

## 🆘 Getting Help

### Documentation
- [Design Summary](DESIGN_SUMMARY_OBSERVABILITY.md)
- [Implementation Details](OBSERVABILITY_IMPLEMENTATION_COMPLETE.md)
- [Observability Checklist](OBSERVABILITY_CHECKLIST.md)
- [Runbooks](runbooks/)

### Common Issues
- [Provider Timeout Runbook](runbooks/provider_timeout.md)
- [Model Load Error Runbook](runbooks/model_load_error.md)

### Support Channels
- GitHub Issues: Report bugs and feature requests
- Slack: #cad-ml-platform
- Email: platform-team@example.com

## 🎯 Next Steps

1. **Customize Dashboards**: Import additional dashboards from Grafana gallery
2. **Set Up Alerts**: Configure alerting rules in Prometheus
3. **Add Custom Metrics**: Extend metrics for your use case
4. **Enable Logging**: Uncomment Loki/Promtail in docker-compose
5. **Production Deployment**: Use Kubernetes manifests for production

---

**Quick Reference Card**

```bash
# Start everything
make obs-up

# Check health
make obs-status

# View metrics
open http://localhost:8000/metrics

# View dashboards
open http://localhost:3000

# Stop everything
make obs-down
```

🎉 **You're all set!** The observability stack is now monitoring your CAD ML Platform.