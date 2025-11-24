# Observability Checklist

## 📋 Pre-Deployment Checklist

### ✅ Metrics Implementation
- [ ] **Core Metrics Exposed**
  - [ ] `ocr_errors_total{provider,code,stage}` - Error tracking with labels
  - [ ] `ocr_input_rejected_total{reason}` - Input validation rejections
  - [ ] `vision_input_rejected_total{reason}` - Vision-specific rejections
  - [ ] `ocr_processing_duration_seconds` - Request latency histogram
  - [ ] `vision_processing_duration_seconds` - Vision latency histogram
  - [ ] `ocr_requests_total` - Total request counter
  - [ ] `vision_requests_total` - Vision request counter

- [ ] **ErrorCode Enum Usage**
  - [ ] All providers use centralized ErrorCode enum
  - [ ] Error mapping abstraction implemented (`error_map.py`)
  - [ ] Consistent error categorization across providers

- [ ] **Metrics Contract Validation**
  - [ ] Run `python -m pytest tests/test_metrics_contract.py`
  - [ ] Run `SELF_CHECK_STRICT_METRICS=1 python scripts/self_check.py`
  - [ ] Verify all required labels present

### ✅ Prometheus Configuration
- [ ] **Recording Rules**
  - [ ] Deploy `docs/prometheus/recording_rules.yml` to Prometheus
  - [ ] Validate with `python scripts/validate_prom_rules.py`
  - [ ] Verify rules loaded: `curl prometheus:9090/api/v1/rules`
  - [ ] Test rule evaluation: `curl prometheus:9090/api/v1/query?query=ocr_error_ratio`

- [ ] **Scrape Configuration**
  ```yaml
  - job_name: 'cad-ml-platform'
    static_configs:
      - targets: ['app:8000']
    scrape_interval: 15s
    scrape_timeout: 10s
  ```

- [ ] **Alerting Rules** (if applicable)
  - [ ] High error rate alert
  - [ ] Provider timeout alert
  - [ ] SLO violation alert

### ✅ Grafana Setup
- [ ] **Dashboard Import**
  - [ ] Import `docs/grafana/observability_dashboard.json`
  - [ ] Configure data source pointing to Prometheus
  - [ ] Verify all panels load correctly
  - [ ] Test time range selection

- [ ] **Key Panels Verification**
  - [ ] Platform Error Rates (Recording Rules)
  - [ ] Provider Error Code Breakdown
  - [ ] Input Rejection Rates
  - [ ] Provider Health Score
  - [ ] Resource Exhaustion Monitoring
  - [ ] Processing Latency P95
  - [ ] OCR Confidence Metrics
  - [ ] SLO Compliance
  - [ ] Error Budget Remaining

### ✅ Operational Tooling
- [ ] **Self-Check Script**
  - [ ] Basic mode: `python scripts/self_check.py`
  - [ ] Strict mode: `SELF_CHECK_STRICT_METRICS=1 python scripts/self_check.py`
  - [ ] JSON output: `python scripts/self_check.py --json`
  - [ ] Remote check: `SELF_CHECK_BASE_URL=https://prod python scripts/self_check.py`

- [ ] **Exit Code Handling**
  - [ ] CI/CD configured to handle exit codes (0, 2, 3, 5, 6)
  - [ ] Appropriate routing for each failure type
  - [ ] Notification channels configured

- [ ] **Security Scanning**
  - [ ] GitHub workflow `.github/workflows/security-audit.yml` enabled
  - [ ] pip-audit configured
  - [ ] bandit scanning active
  - [ ] semgrep rules loaded

### ✅ Documentation
- [ ] **Runbooks Available**
  - [ ] `docs/runbooks/provider_timeout.md` - Timeout response procedures
  - [ ] `docs/runbooks/model_load_error.md` - Model loading issues
  - [ ] Team trained on runbook procedures
  - [ ] Runbooks linked in monitoring alerts

- [ ] **README Updated**
  - [ ] Exit codes table present
  - [ ] Curl examples for error codes
  - [ ] Strict mode documentation
  - [ ] CI/CD examples

- [ ] **Quality Baseline**
  - [ ] `docs/QUALITY_BASELINE.md` updated
  - [ ] Metrics contract documented
  - [ ] ErrorCode values listed
  - [ ] Recording rules documented

## 🚀 Deployment Verification

### Stage 1: Pre-Production
```bash
# 1. Deploy application
kubectl apply -f deployment.yaml

# 2. Wait for readiness
kubectl wait --for=condition=ready pod -l app=cad-ml-platform

# 3. Run self-check
kubectl exec -it deploy/cad-ml-platform -- python scripts/self_check.py

# 4. Verify metrics
kubectl exec -it deploy/cad-ml-platform -- curl localhost:8000/metrics
```

### Stage 2: Metrics Validation
```bash
# 1. Check Prometheus targets
curl prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="cad-ml-platform")'

# 2. Verify metrics ingestion
curl prometheus:9090/api/v1/query?query=up{job="cad-ml-platform"}

# 3. Test recording rules
curl prometheus:9090/api/v1/query?query=ocr_error_ratio

# 4. Validate error tracking
curl prometheus:9090/api/v1/query?query='ocr_errors_total{code="PROVIDER_TIMEOUT"}'
```

### Stage 3: Dashboard Verification
```bash
# 1. Access Grafana
open http://grafana:3000/d/observability

# 2. Check data flow
- Verify all panels show data
- Test time range changes
- Validate drill-downs work

# 3. Test alerts (if configured)
- Trigger test error
- Verify alert fires
- Check notification delivery
```

### Stage 4: Load Testing
```bash
# 1. Generate baseline load
python scripts/load_test.py --rps 10 --duration 300

# 2. Inject errors
python scripts/load_test.py --error-rate 0.1 --timeout-rate 0.05

# 3. Verify metrics reflect errors
curl prometheus:9090/api/v1/query?query='rate(ocr_errors_total[5m])'

# 4. Check dashboards update
- Error rates should increase
- Provider health should decrease
- SLO compliance should change
```

## 📊 Production Monitoring

### Daily Checks
- [ ] Review error rates dashboard
- [ ] Check provider health scores
- [ ] Verify SLO compliance
- [ ] Review error budget consumption

### Weekly Tasks
- [ ] Analyze error patterns
- [ ] Review provider timeout trends
- [ ] Update runbooks if needed
- [ ] Check recording rule performance

### Monthly Review
- [ ] SLO achievement analysis
- [ ] Capacity planning review
- [ ] Error categorization audit
- [ ] Dashboard optimization

## 🔧 Troubleshooting

### Issue: Metrics Not Appearing
```bash
# Check application metrics endpoint
curl localhost:8000/metrics

# Verify Prometheus scraping
curl prometheus:9090/api/v1/targets

# Check for scrape errors
curl prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health=="down")'
```

### Issue: Recording Rules Not Working
```bash
# Reload Prometheus config
curl -X POST prometheus:9090/-/reload

# Check rule evaluation
curl prometheus:9090/api/v1/rules

# Test specific rule
curl prometheus:9090/api/v1/query?query='ocr_error_ratio'
```

### Issue: Dashboard Shows No Data
```bash
# Test data source
curl grafana:3000/api/datasources/proxy/1/api/v1/query?query=up

# Check dashboard variables
- Verify $provider variable populated
- Check time range settings

# Validate queries
- Copy panel query
- Test in Prometheus directly
```

### Issue: Self-Check Failures
```bash
# Debug mode
python scripts/self_check.py 2>&1 | tee self-check.log

# Check specific components
curl localhost:8000/health
curl localhost:8000/metrics | grep ocr_errors_total

# Test with minimal validation
SELF_CHECK_STRICT_METRICS=0 python scripts/self_check.py
```

## 📈 Success Metrics

### Observability KPIs
- **Metric Coverage**: >95% of errors tracked with ErrorCode
- **Dashboard Load Time**: <3 seconds
- **Alert Response Time**: <5 minutes
- **Runbook Effectiveness**: >80% issues resolved using runbooks
- **Self-Check Pass Rate**: >99% in production

### Quality Gates
- **Pre-Deployment**: Self-check must pass (exit code 0)
- **Canary**: Error rate <1% increase
- **Production**: SLO compliance >99.5%
- **Rollback Trigger**: Error budget exhaustion

## 🎯 Phase 2 Readiness

### Confirmed Ready for Refactoring
- [ ] Metrics baseline established
- [ ] Error patterns documented
- [ ] Performance benchmarks recorded
- [ ] Runbooks tested and validated
- [ ] Team trained on observability tools

### Migration Safety Net
- [ ] Recording rules capturing current state
- [ ] Dashboards showing historical trends
- [ ] Alerts configured for regression detection
- [ ] Rollback procedures documented

## 📝 Sign-off

- [ ] **Engineering Lead**: Metrics implementation complete
- [ ] **DevOps Lead**: Monitoring stack configured
- [ ] **QA Lead**: Testing procedures validated
- [ ] **Product Manager**: KPIs and SLOs approved

---

*Last Updated*: 2025-01-20
*Version*: 1.0.0
*Next Review*: 2025-02-20