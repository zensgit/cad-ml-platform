# 🎓 Training Guide - CAD ML Platform Observability

## Welcome to the Training Program

This comprehensive training guide will help you master the CAD ML Platform's observability infrastructure. Whether you're an operator, developer, or SRE, this guide provides structured learning paths to build your expertise.

## 📚 Learning Paths

### Path 1: Operations Team (8 hours)
Perfect for team members responsible for daily monitoring and incident response.

### Path 2: Development Team (6 hours)
For developers who need to instrument code and understand metrics.

### Path 3: SRE/DevOps Team (10 hours)
Comprehensive training for infrastructure and reliability engineering.

---

## 🎯 Module 1: System Overview (2 hours)

### Learning Objectives
- Understand the CAD ML Platform architecture
- Identify key components and their interactions
- Navigate the monitoring infrastructure
- Access and use primary dashboards

### 1.1 Platform Architecture

#### Component Map
```
Frontend (React) → API Gateway → Services → Providers
                         ↓
                    Metrics/Logs
                         ↓
                 Observability Stack
```

#### Key Services
1. **OCR Service**: Handles document text extraction
2. **Vision Service**: Processes images and visual analysis
3. **Assembly Service**: Manages mechanical assembly understanding
4. **API Gateway**: Routes and load balances requests

### 1.2 Observability Stack

#### Core Components
```yaml
Monitoring:
  Prometheus: Time-series metrics database
  Grafana: Visualization and dashboards
  AlertManager: Alert routing and management

Logging:
  Loki: Log aggregation
  Promtail: Log shipping

Supporting:
  Redis: Caching and rate limiting
  Node Exporter: System metrics
```

### 1.3 Hands-On Exercise

**Exercise 1.1: Dashboard Navigation**
1. Access Grafana at http://localhost:3000
2. Navigate to "Platform Overview" dashboard
3. Identify these metrics:
   - Current error rate
   - Request throughput
   - P95 latency
4. Change time range to last 24 hours
5. Create a custom panel showing provider health

**Exercise 1.2: Metric Exploration**
1. Access Prometheus at http://localhost:9090
2. Run these queries:
```promql
# Total requests
sum(rate(ocr_requests_total[5m]))

# Error ratio
platform_error_ratio

# Provider health
provider_health_score
```

### 1.4 Knowledge Check
- [ ] Can you explain the data flow from service to dashboard?
- [ ] What is the difference between Prometheus and Grafana?
- [ ] How would you check if a specific provider is healthy?

---

## 🔍 Module 2: Metrics and Monitoring (2 hours)

### Learning Objectives
- Understand metric types and naming conventions
- Write Prometheus queries
- Create custom dashboards
- Interpret recording rules

### 2.1 Metrics Fundamentals

#### Metric Types
```yaml
Counter:
  Description: Cumulative values that only increase
  Example: ocr_requests_total
  Query: rate(ocr_requests_total[5m])

Gauge:
  Description: Values that can go up or down
  Example: concurrent_requests
  Query: concurrent_requests

Histogram:
  Description: Observations in buckets
  Example: ocr_processing_duration_seconds
  Query: histogram_quantile(0.95, rate(ocr_processing_duration_seconds_bucket[5m]))
```

### 2.2 Metrics Contract

#### Required Metrics
```python
REQUIRED_METRICS = {
    # Error tracking
    "ocr_errors_total": {
        "type": "counter",
        "labels": ["provider", "code", "stage"],
        "description": "Total OCR errors by provider and error code"
    },

    # Input validation
    "ocr_input_rejected_total": {
        "type": "counter",
        "labels": ["reason"],
        "description": "Rejected inputs with reason"
    },

    # Performance
    "ocr_processing_duration_seconds": {
        "type": "histogram",
        "labels": ["provider", "stage"],
        "description": "Processing time distribution"
    }
}
```

### 2.3 Writing PromQL Queries

#### Basic Queries
```promql
# Request rate per provider
sum by(provider) (rate(ocr_requests_total[5m]))

# Error percentage
100 * (
  sum(rate(ocr_errors_total[5m]))
  /
  sum(rate(ocr_requests_total[5m]))
)

# P95 latency per provider
histogram_quantile(0.95,
  sum by(provider, le) (
    rate(ocr_processing_duration_seconds_bucket[5m])
  )
)
```

#### Advanced Queries
```promql
# Week-over-week comparison
sum(rate(ocr_requests_total[5m]))
/
sum(rate(ocr_requests_total[5m] offset 7d))

# Alert condition
(sum(rate(ocr_errors_total[5m])) > 0.1)
and
(sum(rate(ocr_requests_total[5m])) > 1)
```

### 2.4 Recording Rules

#### Purpose and Benefits
- Pre-calculate expensive queries
- Improve dashboard performance
- Standardize calculations

#### Example Rules
```yaml
- record: platform_error_ratio
  expr: |
    100 * (
      sum(rate(ocr_errors_total[5m]))
      /
      sum(rate(ocr_requests_total[5m]))
    )

- record: provider_health_score
  expr: |
    100 * (1 -
      (sum by(provider) (rate(ocr_errors_total[5m])))
      /
      (sum by(provider) (rate(ocr_requests_total[5m])))
    )
```

### 2.5 Hands-On Exercise

**Exercise 2.1: Query Writing**
Write PromQL queries to answer:
1. Which provider has the highest error rate?
2. What's the 99th percentile latency?
3. How many requests failed in the last hour?

**Exercise 2.2: Dashboard Creation**
1. Create a new dashboard in Grafana
2. Add panels for:
   - Request rate (line graph)
   - Error distribution (pie chart)
   - Latency heatmap
3. Add variables for provider selection
4. Set up auto-refresh

### 2.6 Knowledge Check
- [ ] What's the difference between rate() and increase()?
- [ ] When would you use a recording rule?
- [ ] How do you calculate percentiles from histograms?

---

## 🚨 Module 3: Alerting and Incident Response (3 hours)

### Learning Objectives
- Configure and manage alerts
- Understand alert routing
- Execute incident response procedures
- Perform root cause analysis

### 3.1 Alert Configuration

#### Alert Structure
```yaml
alert: HighErrorRate
expr: platform_error_ratio > 5  # Condition
for: 5m                         # Duration before firing
labels:
  severity: critical            # Routing label
  component: platform
annotations:
  summary: "High error rate"
  description: "Error rate is {{ $value }}%"
  runbook_url: "https://runbook.link"
```

#### Severity Levels
```yaml
Critical:
  Response: Immediate (page on-call)
  Examples: Platform down, data loss
  Routing: PagerDuty + Slack + Email

Warning:
  Response: 30 minutes
  Examples: Degraded performance
  Routing: Slack

Info:
  Response: Next business day
  Examples: Capacity planning
  Routing: Email only
```

### 3.2 Alert Routing

#### AlertManager Configuration
```yaml
route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true
```

### 3.3 Incident Response Process

#### Response Framework
```
1. DETECT → Alert fires
2. ACKNOWLEDGE → Responder acknowledges
3. ASSESS → Determine impact and scope
4. COMMUNICATE → Update stakeholders
5. MITIGATE → Apply immediate fix
6. RESOLVE → Implement permanent solution
7. REVIEW → Post-incident analysis
```

#### Runbook Template
```markdown
## Alert: [Alert Name]

### Symptoms
- What the user experiences
- What metrics show

### Impact
- Services affected
- User impact
- Business impact

### Diagnosis
1. Check dashboard X
2. Run query Y
3. Review logs Z

### Mitigation
1. Immediate action
2. Temporary fix
3. Verify resolution

### Resolution
1. Root cause fix
2. Validation steps
3. Monitoring confirmation
```

### 3.4 Root Cause Analysis

#### Investigation Tools
```bash
# Check recent errors
curl http://localhost:8000/metrics | grep error

# View provider status
./scripts/self_check.py --json | jq '.providers'

# Analyze logs
kubectl logs -n cad-ml-platform deployment/cad-ml-platform --tail=100

# Check recent changes
kubectl rollout history deployment/cad-ml-platform
```

### 3.5 Hands-On Exercise

**Exercise 3.1: Alert Simulation**
1. Trigger a test alert:
```bash
curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot
```
2. Observe alert flow in AlertManager
3. Practice acknowledgment
4. Document response steps

**Exercise 3.2: Incident Response Drill**
Scenario: High error rate alert fired
1. Acknowledge alert
2. Identify affected provider
3. Check circuit breaker status
4. Review recent deployments
5. Implement mitigation
6. Write incident report

### 3.6 Knowledge Check
- [ ] What's the difference between group_wait and group_interval?
- [ ] How do you prevent alert fatigue?
- [ ] What are the key steps in incident response?

---

## 🛠️ Module 4: Operational Procedures (2 hours)

### Learning Objectives
- Execute daily operational tasks
- Perform health checks
- Manage deployments
- Maintain system hygiene

### 4.1 Daily Operations

#### Morning Checklist (9 AM)
```bash
# 1. Check overnight alerts
open http://localhost:9093  # AlertManager

# 2. Review dashboard
open http://localhost:3000/d/platform-overview

# 3. Run health check
./scripts/self_check.py --strict

# 4. Check resource usage
kubectl top pods -n cad-ml-platform

# 5. Review error logs
kubectl logs -n cad-ml-platform -l app=cad-ml-platform --since=12h | grep ERROR
```

#### Continuous Monitoring
```yaml
Every Hour:
  - Check error rate (<1%)
  - Verify latency (<2s P95)
  - Monitor queue depth
  - Review active alerts

Every 4 Hours:
  - Check provider health scores
  - Review resource trends
  - Validate backups
```

### 4.2 Health Validation

#### Using Self-Check Script
```bash
# Basic health check
./scripts/self_check.py

# Strict validation (for CI/CD)
./scripts/self_check.py --strict

# JSON output for automation
./scripts/self_check.py --json

# Exit codes
# 0: Success
# 2: API failure
# 3: Health check failed
# 5: Contract violation
# 6: Provider mapping gap
```

#### Manual Validation
```bash
# Service health
curl http://localhost:8000/health

# Metrics availability
curl http://localhost:8000/metrics

# Provider status
curl http://localhost:8000/health/providers
```

### 4.3 Deployment Management

#### Pre-Deployment
```bash
# 1. Validate configuration
make validate

# 2. Run tests
make test
# Optional: Faiss perf tests are gated (see README.md / docs/OPERATIONAL_RUNBOOK.md)
# RUN_FAISS_PERF_TESTS=1 pytest tests/perf/test_vector_search_latency.py -v

# 3. Check current state
kubectl get all -n cad-ml-platform

# 4. Backup current config
kubectl get deployment -o yaml > backup.yaml
```

#### Deployment Process
```bash
# 1. Build and push image
make docker-build docker-push

# 2. Deploy to staging
./scripts/deploy_production.sh --environment staging

# 3. Validate staging
make staging-smoke-test

# 4. Deploy to production
./scripts/deploy_production.sh --environment production

# 5. Monitor metrics
watch -n 5 './scripts/self_check.py --json | jq .summary'
```

#### Post-Deployment
```bash
# 1. Verify deployment
kubectl rollout status deployment/cad-ml-platform

# 2. Check metrics
curl http://localhost:8000/metrics | grep -E "ocr_requests|errors"

# 3. Monitor for 30 minutes
# Watch dashboards for anomalies

# 4. If issues, rollback
kubectl rollout undo deployment/cad-ml-platform
```

### 4.4 Maintenance Tasks

#### Weekly Tasks
```yaml
Monday:
  - Review week's incidents
  - Update runbooks
  - Check backup integrity

Wednesday:
  - Performance review
  - Capacity planning
  - Alert tuning

Friday:
  - Security scan
  - Dependency updates
  - Documentation review
```

#### Monthly Tasks
```yaml
First Monday:
  - Full backup test
  - Disaster recovery drill
  - Compliance audit

Mid-Month:
  - Performance optimization
  - Cost review
  - Training updates
```

### 4.5 Hands-On Exercise

**Exercise 4.1: Daily Operations**
1. Complete morning checklist
2. Document any findings
3. Create status report

**Exercise 4.2: Deployment Simulation**
1. Deploy to staging environment
2. Run validation tests
3. Monitor metrics
4. Practice rollback

### 4.6 Knowledge Check
- [ ] What are the key daily checks?
- [ ] How do you validate a deployment?
- [ ] When would you trigger a rollback?

---

## 🔧 Module 5: Troubleshooting (2 hours)

### Learning Objectives
- Diagnose common issues
- Use debugging tools effectively
- Optimize performance
- Resolve provider-specific problems

### 5.1 Common Issues

#### Issue: High Error Rate
```bash
# Step 1: Identify error patterns
curl http://localhost:8000/metrics | grep errors_total

# Step 2: Check specific error codes
promql: sum by(code) (rate(ocr_errors_total[5m]))

# Step 3: Review provider health
./scripts/self_check.py --json | jq '.providers'

# Step 4: Check recent changes
git log --oneline -10
kubectl rollout history deployment/cad-ml-platform

# Step 5: Review logs for stack traces
kubectl logs deployment/cad-ml-platform | grep -A 10 ERROR
```

#### Issue: High Latency
```bash
# Step 1: Measure current latency
promql: histogram_quantile(0.95, rate(ocr_processing_duration_seconds_bucket[5m]))

# Step 2: Identify slow providers
promql: histogram_quantile(0.95, sum by(provider, le) (rate(ocr_processing_duration_seconds_bucket[5m])))

# Step 3: Check resource constraints
kubectl top pods
kubectl describe pod <pod-name>

# Step 4: Review circuit breaker status
curl http://localhost:8000/health/circuit-breakers

# Step 5: Analyze slow queries
# Check database query logs
```

#### Issue: Provider Failures
```bash
# Step 1: Check provider status
curl https://status.provider.com

# Step 2: Test provider directly
curl -X POST https://api.provider.com/test \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"test": "data"}'

# Step 3: Review provider-specific errors
promql: sum by(provider, code) (rate(ocr_errors_total{provider="deepseek"}[5m]))

# Step 4: Check rate limits
grep -i "rate.limit\|quota" logs/app.log

# Step 5: Verify credentials
echo $PROVIDER_API_KEY | base64 -d
```

### 5.2 Debugging Tools

#### Log Analysis
```bash
# Structured log search
kubectl logs deployment/cad-ml-platform | jq '. | select(.level=="ERROR")'

# Time-based filtering
kubectl logs deployment/cad-ml-platform --since=1h

# Pattern matching
kubectl logs deployment/cad-ml-platform | grep -E "timeout|error|fail"

# Follow logs in real-time
kubectl logs -f deployment/cad-ml-platform
```

#### Metrics Deep Dive
```promql
# Compare current vs historical
rate(ocr_requests_total[5m]) / rate(ocr_requests_total[5m] offset 1d)

# Identify anomalies
abs(rate(ocr_requests_total[5m]) - avg_over_time(rate(ocr_requests_total[5m])[1h:5m])) > 2 * stddev_over_time(rate(ocr_requests_total[5m])[1h:5m])

# Correlation analysis
# High errors when high load?
rate(ocr_errors_total[5m]) / rate(ocr_requests_total[5m])
```

### 5.3 Performance Optimization

#### Query Optimization
```yaml
Before:
  Query: sum(rate(ocr_requests_total[5m])) by (provider)
  Latency: 200ms

After (with recording rule):
  Query: requests_per_provider
  Latency: 20ms

Recording Rule:
  - record: requests_per_provider
    expr: sum(rate(ocr_requests_total[5m])) by (provider)
```

#### Resource Tuning
```yaml
# CPU optimization
resources:
  requests:
    cpu: "500m"  # Start conservative
  limits:
    cpu: "2000m" # Allow bursting

# Memory optimization
resources:
  requests:
    memory: "512Mi"
  limits:
    memory: "2Gi"

# HPA tuning
spec:
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### 5.4 Advanced Diagnostics

#### Distributed Tracing
```bash
# Enable tracing
export TRACING_ENABLED=true
export JAEGER_ENDPOINT=http://jaeger:14268

# Query traces
curl http://jaeger:16686/api/traces?service=cad-ml-platform

# Analyze slow spans
# Look for operations taking >100ms
```

#### Profiling
```bash
# CPU profiling
curl http://localhost:8000/debug/pprof/profile?seconds=30 > cpu.prof
go tool pprof cpu.prof

# Memory profiling
curl http://localhost:8000/debug/pprof/heap > heap.prof
go tool pprof heap.prof

# Goroutine analysis
curl http://localhost:8000/debug/pprof/goroutine?debug=2
```

### 5.5 Hands-On Exercise

**Exercise 5.1: Troubleshooting Scenario**
Scenario: Users report slow OCR processing
1. Identify affected providers
2. Measure current latency
3. Check resource usage
4. Review error logs
5. Propose solution

**Exercise 5.2: Performance Tuning**
1. Identify slow dashboard queries
2. Create recording rule
3. Measure improvement
4. Document findings

### 5.6 Knowledge Check
- [ ] How do you identify which provider is failing?
- [ ] What tools help with performance analysis?
- [ ] When should you create a recording rule?

---

## 🎯 Module 6: Advanced Topics (2 hours)

### Learning Objectives
- Implement SLOs and error budgets
- Configure advanced alerting
- Optimize costs
- Plan for scale

### 6.1 SLO Implementation

#### Defining SLOs
```yaml
SLO_Definition:
  Service: CAD ML Platform
  Metric: Availability
  Target: 99.5%
  Window: 30 days

  Calculation: |
    1 - (failed_requests / total_requests)

  Error_Budget: |
    0.5% = 216 minutes/month
```

#### SLO Monitoring
```promql
# Current availability
1 - (
  sum(rate(ocr_errors_total[30d])) /
  sum(rate(ocr_requests_total[30d]))
)

# Error budget consumed
(1 - slo_availability) / 0.005 * 100

# Burn rate
rate(slo_errors_total[1h]) / (0.005 / 720)
```

### 6.2 Advanced Alerting

#### Multi-Window Alerts
```yaml
- alert: ErrorBudgetBurnRate
  expr: |
    (
      rate(slo_errors_total[1h]) > (14.4 * 0.005/720)
    ) and (
      rate(slo_errors_total[5m]) > (14.4 * 0.005/720)
    )
  labels:
    severity: critical
  annotations:
    summary: "Error budget burn rate too high"
```

#### Predictive Alerts
```promql
# Predict disk full in 4 hours
predict_linear(node_filesystem_free_bytes[1h], 4*3600) < 0

# Predict memory exhaustion
predict_linear(container_memory_usage_bytes[10m], 3600) > container_spec_memory_limit_bytes
```

### 6.3 Cost Optimization

#### Metrics Retention
```yaml
Retention_Strategy:
  Raw_Metrics: 15 days
  5m_Aggregates: 30 days
  1h_Aggregates: 90 days
  1d_Aggregates: 365 days

Implementation:
  - Use recording rules for aggregation
  - Implement downsampling
  - Archive to object storage
```

#### Resource Optimization
```bash
# Identify over-provisioned pods
kubectl top pods | awk '$3<20 {print $1, $3"%"}'

# Find unused metrics
promql: count by(__name__)({__name__=~".+"})

# Optimize cardinality
promql: count by(__name__)(count by(__name__, job)({__name__=~".+"}))
```

### 6.4 Scaling Strategies

#### Horizontal Scaling
```yaml
HPA_Config:
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: request_rate
        target:
          type: AverageValue
          averageValue: "1000"
```

#### Sharding Strategy
```yaml
Prometheus_Sharding:
  Shard_1: Services A-M
  Shard_2: Services N-Z

  Federation:
    Global_Prometheus: Aggregates from shards

  Benefits:
    - Reduced query latency
    - Better resource utilization
    - Failure isolation
```

### 6.5 Future Roadmap Skills

#### Emerging Technologies
1. **OpenTelemetry**: Unified observability framework
2. **eBPF**: Kernel-level observability
3. **AI/ML Monitoring**: Anomaly detection, predictive analytics
4. **Edge Observability**: Distributed edge monitoring

### 6.6 Hands-On Exercise

**Exercise 6.1: SLO Configuration**
1. Define SLO for your service
2. Create error budget alert
3. Build SLO dashboard
4. Test budget consumption

**Exercise 6.2: Cost Analysis**
1. Calculate current metrics storage
2. Identify high-cardinality metrics
3. Propose optimization plan
4. Estimate cost savings

---

## 📖 Quick Reference Guide

### Essential Commands

#### Health Checks
```bash
# Platform health
./scripts/self_check.py --strict --json

# Prometheus targets
curl http://localhost:9090/api/v1/targets

# Provider status
curl http://localhost:8000/health/providers
```

#### Debugging
```bash
# Recent errors
kubectl logs -l app=cad-ml-platform --tail=50 | grep ERROR

# Metrics check
curl -s http://localhost:8000/metrics | grep -E "errors|latency"

# Alert status
curl http://localhost:9093/api/v1/alerts
```

#### Deployment
```bash
# Deploy staging
./scripts/deploy_production.sh --environment staging

# Rollback
kubectl rollout undo deployment/cad-ml-platform

# Scale
kubectl scale deployment/cad-ml-platform --replicas=5
```

### Key Metrics Queries

```promql
# Error rate
100 * sum(rate(ocr_errors_total[5m])) / sum(rate(ocr_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, sum(rate(ocr_processing_duration_seconds_bucket[5m])) by (le))

# Provider health
100 * (1 - (sum by(provider) (rate(ocr_errors_total[5m])) / sum by(provider) (rate(ocr_requests_total[5m]))))

# Request rate
sum(rate(ocr_requests_total[5m]))
```

### Alert Priorities

| Severity | Response Time | Examples |
|----------|--------------|----------|
| P1 | 15 min | Platform down |
| P2 | 30 min | Feature broken |
| P3 | 2 hours | Performance degraded |
| P4 | 24 hours | Minor issues |

---

## 🎓 Certification Path

### Level 1: Operator Certification
- [ ] Complete Modules 1-4
- [ ] Pass knowledge checks
- [ ] Handle 5 practice incidents
- [ ] Score 80% on assessment

### Level 2: Advanced Operator
- [ ] Complete all modules
- [ ] Create custom dashboard
- [ ] Write 3 runbooks
- [ ] Lead incident response

### Level 3: Platform Expert
- [ ] Design monitoring strategy
- [ ] Implement SLOs
- [ ] Optimize performance
- [ ] Train new team members

---

## 📚 Additional Resources

### Documentation
- Operations Manual: `docs/OPERATIONS_MANUAL.md`
- API Documentation: `docs/API.md`
- Runbook Library: `docs/runbooks/`

### External Training
- Prometheus Training: https://prometheus.io/docs/tutorials/
- Grafana University: https://grafana.com/tutorials/
- Kubernetes Basics: https://kubernetes.io/docs/tutorials/

### Community
- Slack: #cad-ml-platform
- Wiki: https://wiki.example.com/cad-ml
- Office Hours: Thursdays 2-3 PM

---

## 🎯 Training Completion

Congratulations on completing the CAD ML Platform Observability Training! You now have the knowledge and skills to:

- Monitor and maintain the platform
- Respond to incidents effectively
- Optimize performance
- Troubleshoot issues
- Plan for scale

Remember: Observability is a journey, not a destination. Continue learning, experimenting, and improving!

**Next Steps:**
1. Practice with real scenarios
2. Shadow experienced operators
3. Contribute to runbooks
4. Share knowledge with team

---

**Training Version**: 1.0.0
**Last Updated**: January 21, 2025
**Feedback**: training@example.com
