# Evaluation System Maintenance Guide

## Quick Reference

### Daily Operations
```bash
# Run evaluation and generate all reports
make eval-e2e

# Quick health check
make eval-combined-save && make eval-insights-json
```

### Weekly Maintenance
```bash
# Apply retention policy (automated in CI)
make eval-retention-apply

# Update anomaly baseline
make baseline-update

# Review anomaly trends
python3 scripts/analyze_eval_insights.py --days 7
```

### Monthly Reviews
```bash
# Review and adjust badge thresholds
vim config/eval_frontend.json  # Check score_thresholds

# Generate monthly report
python3 scripts/analyze_eval_insights.py --days 30 --output reports/monthly_$(date +%Y%m).md

# Security audit
make security-audit
```

### Quarterly Tasks
```bash
# Snapshot baseline for archival
cp reports/insights/baseline.json reports/baselines/baseline_$(date +%Y_Q%q).json

# Deep security review
python3 scripts/security_audit.py --severity low --json > reports/security_Q$(date +%q).json

# Performance analysis
python3 scripts/export_eval_metrics.py --format json --output reports/metrics_Q$(date +%q).json
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. SHA-384 Integrity Check Failures
**Symptom**: `integrity-check-strict` fails with SHA mismatch
```bash
# Update SHA in config after verifying Chart.js version
sha384sum path/to/chart.js
# Update config/eval_frontend.json with new hash
```

#### 2. Anomaly Detection False Positives
**Symptom**: Too many anomalies reported on stable system
```bash
# Check baseline health
python3 scripts/anomaly_baseline.py --show

# If sample count < 30, wait for more data
# If volatility is high, adjust threshold
vim scripts/anomaly_baseline.py  # Increase outlier_threshold from 2.0 to 2.5
```

#### 3. Metrics Server Port Conflicts
**Symptom**: "Address already in use" error
```bash
# Find what's using the port
lsof -i :8000
# Or
netstat -an | grep 8000

# Use different port
make metrics-serve PORT=8001
```

#### 4. CI Exit Code Routing
**Based on security_audit.py exit codes:**
```yaml
# GitHub Actions example
- name: Security Audit
  id: security
  continue-on-error: true
  run: python3 scripts/security_audit.py --severity high

- name: Route based on exit code
  if: steps.security.outcome == 'failure'
  run: |
    case "${{ steps.security.outputs.exit-code }}" in
      2) echo "::error::Critical vulnerabilities - notify security team" ;;
      3) echo "::error::Secrets exposed - immediate rotation required" ;;
      4) echo "::warning::High severity deps - schedule update" ;;
      5) echo "::warning::Docker issues - review container security" ;;
      6) echo "::warning::Code security - review with dev team" ;;
    esac
```

## Performance Optimization

### Token Usage Monitoring
```bash
# Check evaluation file sizes
du -sh reports/eval_history/*.json | tail -20

# If files > 100KB, consider compression
find reports/eval_history -name "*.json" -size +100k -exec gzip {} \;
```

### Baseline Cache Management
```bash
# Check baseline size
ls -lh reports/insights/baseline.json

# If > 1MB, consider trimming history
python3 scripts/anomaly_baseline.py --reset
python3 scripts/anomaly_baseline.py --update  # Rebuild from recent data
```

### Metrics Export Optimization
```bash
# For high-frequency scraping, use server mode
make metrics-serve &  # Run in background

# For batch processing, use file export
make metrics-export OUTPUT=reports/metrics_$(date +%s).prom
```

## Configuration Tuning

### Badge Color Thresholds
Edit `config/eval_frontend.json`:
```json
{
  "badge_thresholds": {
    "excellent": 0.9,    // brightgreen
    "good": 0.8,        // green
    "acceptable": 0.7,   // yellow
    "warning": 0.6,     // orange
    "critical": 0.0     // red
  }
}
```

### Anomaly Detection Sensitivity
Edit `scripts/anomaly_baseline.py`:
```python
"config": {
    "max_history_size": 100,        # Increase for more stability
    "min_samples_for_baseline": 10, # Increase to reduce noise
    "outlier_threshold": 2.0        # Increase to reduce false positives
}
```

### Retention Policy
Edit `config/eval_frontend.json`:
```json
{
  "retention_policy": {
    "active_days": 7,
    "archive_30d_days": 30,
    "archive_90d_days": 90,
    "archive_365d_days": 365
  }
}
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cad_ml_evaluation'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5m
    scrape_timeout: 30s
```

### Grafana Dashboard Queries
```promql
# Combined score over time
cad_ml_evaluation_score{module="combined"}

# Vision vs OCR comparison
cad_ml_evaluation_score{module=~"vision|ocr"}

# Anomaly detection (requires recording rule)
rate(cad_ml_evaluation_score{module="combined"}[5m]) > 0.1
```

### Alert Rules
```yaml
# prometheus/alerts.yml
groups:
  - name: cad_ml_alerts
    rules:
      - alert: EvaluationScoreDrop
        expr: cad_ml_evaluation_score{module="combined"} < 0.6
        for: 10m
        annotations:
          summary: "Combined score below threshold"

      - alert: SecurityVulnerabilities
        expr: cad_ml_security_critical > 0
        for: 1m
        annotations:
          summary: "Critical security vulnerabilities detected"
```

## Future Enhancement Tracking

### Predictive Modeling (Planned)
```python
# Concept for forecast.json generation
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def generate_forecast():
    # Load historical data
    history = load_evaluation_history()

    # Fit ARIMA model
    model = ARIMA(history['combined_score'], order=(1,1,1))
    fitted = model.fit()

    # Generate 7-day forecast
    forecast = fitted.forecast(steps=7)

    # Save to forecast.json
    save_forecast(forecast)
```

### Adaptive Thresholds (Planned)
```python
# Dynamic threshold adjustment based on volatility
def calculate_adaptive_threshold(history):
    recent_volatility = history[-30:].std()
    baseline_volatility = history.std()

    if recent_volatility > baseline_volatility * 1.5:
        # High volatility period - widen bands
        return 2.5
    else:
        # Stable period - tighten bands
        return 2.0
```

### Multi-Repo Federation (Planned)
```yaml
# federation_config.yaml
repositories:
  - name: cad-ml-platform
    eval_endpoint: http://cad-ml:8000/metrics
    weight: 1.0

  - name: ocr-service
    eval_endpoint: http://ocr:8000/metrics
    weight: 0.5

  - name: vision-service
    eval_endpoint: http://vision:8000/metrics
    weight: 0.5
```

### Lightweight WebUI (Planned)
```html
<!-- Concept for minimal SPA -->
<!DOCTYPE html>
<html>
<head>
    <title>CAD ML Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div id="badges"></div>
    <canvas id="trend-chart"></canvas>
    <div id="anomalies"></div>

    <script>
        // Fetch and display evaluation data
        fetch('/reports/eval_history/latest_combined.json')
            .then(r => r.json())
            .then(data => renderDashboard(data));
    </script>
</body>
</html>
```

## Best Practices

### 1. Evaluation Consistency
- Run evaluations at consistent times (e.g., every 6 hours)
- Use the same test dataset for trending
- Document any dataset changes in git commits

### 2. Alert Fatigue Prevention
- Start with conservative thresholds
- Gradually tighten based on observed patterns
- Use anomaly baseline to filter noise

### 3. Security Response
- Treat exit code 3 (secrets) as P0 emergency
- Exit code 2 (critical vulns) requires same-day response
- Schedule monthly dependency updates

### 4. Data Hygiene
- Run retention policy weekly
- Archive quarterly baselines
- Compress old evaluation files

### 5. Documentation
- Update this guide when thresholds change
- Document anomaly investigations
- Keep runbooks for common issues

## Quick Commands Cheatsheet

```bash
# Health Check
make eval-combined-save eval-insights-json | jq '.risk_level'

# Anomaly Investigation
python3 scripts/analyze_eval_insights.py --days 1 --verbose

# Security Status
python3 scripts/security_audit.py --severity critical --json | jq '.summary'

# Metrics Snapshot
curl -s localhost:8000/metrics | grep "^cad_ml_evaluation_score"

# Badge Status
python3 scripts/generate_badge.py --format json | jq '.status'

# Baseline Health
python3 scripts/anomaly_baseline.py --show | jq '.metrics.combined'

# Quick Trend
make eval-trend DAYS=7

# Full Workflow Test
time make eval-phase6
```

## Support Matrix

| Component | Version | Required | Notes |
|-----------|---------|----------|-------|
| Python | 3.11+ | Yes | Core runtime |
| Node.js | 18+ | Optional | For npm audit |
| Docker | 20+ | Optional | For trivy scanning |
| Make | 3.81+ | Yes | Workflow orchestration |
| Git | 2.25+ | Yes | Version control |
| curl | 7.68+ | Yes | Metrics testing |
| jq | 1.6+ | Recommended | JSON processing |

## Contact & Escalation

### Issue Types and Owners
- **Evaluation Logic**: Data Science Team
- **Infrastructure**: Platform Team
- **Security Issues**: Security Team (exit codes 2-3)
- **CI/CD Pipeline**: DevOps Team
- **Documentation**: Technical Writing Team

### Escalation Path
1. Check this maintenance guide
2. Review `docs/PHASE*_*.md` documentation
3. Check GitHub issues for known problems
4. Create new issue with:
   - Error messages/exit codes
   - `make eval-validate` output
   - Last known good evaluation timestamp
   - Recent changes to codebase

---

*Last Updated: 2025-11-19*
*Version: 1.0.0*
*Next Review: End of Q1 2025*