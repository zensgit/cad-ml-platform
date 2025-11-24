# Runbook: PROVIDER_TIMEOUT

## Overview
This runbook addresses OCR provider timeout errors that occur when a provider fails to respond within the configured time limit.

## Error Code
- **Code**: `PROVIDER_TIMEOUT`
- **Severity**: High
- **Impact**: Service degradation, reduced throughput, potential request failures

## Detection

### Metrics
```promql
# High timeout rate for specific provider
rate(ocr_errors_total{code="PROVIDER_TIMEOUT", provider="$provider"}[5m]) > 0.1

# Provider timeout spike
increase(ocr_errors_total{code="PROVIDER_TIMEOUT"}[5m]) > 10

# Recording rule alert
provider_timeout_rate{provider="$provider"} > 0.5
```

### Logs
```bash
# Check application logs
grep -i "PROVIDER_TIMEOUT" /app/logs/app.log | tail -20

# Check provider-specific logs
grep -i "timeout" /app/logs/ocr_$PROVIDER.log | tail -50

# Check network timeouts
journalctl -u cad-ml-platform --since "5 minutes ago" | grep -i timeout
```

### Dashboard
- Grafana Dashboard: "CAD ML Platform - Observability"
- Panel: "Provider Timeout Rates"
- Look for spikes above baseline (normal: <0.01/sec)

## Response Steps

### 1. Immediate Actions (0-5 minutes)

#### Verify the issue
```bash
# Check current timeout rate
curl http://localhost:8000/metrics | grep provider_timeout

# Test provider directly
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{"provider": "$PROVIDER", "test": true}'

# Check circuit breaker status
curl http://localhost:8000/health | jq '.providers.$PROVIDER.circuit_breaker'
```

#### Isolate affected provider
```bash
# If circuit breaker hasn't tripped, manually disable provider
export DISABLED_PROVIDERS="$PROVIDER"
# Or update config
echo "disabled_providers: [$PROVIDER]" >> /app/config/overrides.yaml
systemctl reload cad-ml-platform
```

### 2. Investigation (5-15 minutes)

#### Check provider health
```python
# Test provider connectivity
import asyncio
from src.core.ocr.providers import get_provider

async def test_provider(name):
    provider = get_provider(name)
    try:
        result = await asyncio.wait_for(
            provider.health_check(),
            timeout=5
        )
        print(f"Provider {name}: {result}")
    except asyncio.TimeoutError:
        print(f"Provider {name}: TIMEOUT")

asyncio.run(test_provider("$PROVIDER"))
```

#### Analyze timeout patterns
```sql
-- Query metrics database (if using)
SELECT
    date_trunc('minute', timestamp) as minute,
    count(*) as timeout_count,
    avg(duration_ms) as avg_duration
FROM ocr_requests
WHERE provider = '$PROVIDER'
    AND error_code = 'PROVIDER_TIMEOUT'
    AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY minute
ORDER BY minute DESC;
```

#### Check resource usage
```bash
# Memory pressure
free -h
vmstat 1 5

# CPU usage
top -b -n 1 | head -20

# Network latency
ping -c 5 provider.endpoint.com
traceroute provider.endpoint.com

# Connection pool exhaustion
netstat -an | grep ESTABLISHED | wc -l
ss -tan | grep TIME-WAIT | wc -l
```

### 3. Mitigation (15-30 minutes)

#### Adjust timeout settings
```yaml
# config/providers.yaml
providers:
  $PROVIDER:
    timeout_seconds: 30  # Increase from default 10
    retry_attempts: 3
    retry_delay_ms: 1000
```

#### Scale provider resources
```bash
# If using containerized providers
kubectl scale deployment ocr-provider-$PROVIDER --replicas=3

# Increase resource limits
kubectl set resources deployment ocr-provider-$PROVIDER \
  --limits=cpu=2,memory=4Gi \
  --requests=cpu=1,memory=2Gi
```

#### Implement fallback
```python
# Enable provider fallback
from src.core.ocr import OCROrchestrator

orchestrator = OCROrchestrator()
orchestrator.enable_fallback({
    "$PROVIDER": ["deepseek", "tesseract"]  # Fallback providers
})
```

### 4. Recovery (30-60 minutes)

#### Re-enable provider gradually
```bash
# Test with single request
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{"provider": "$PROVIDER", "test": true}'

# If successful, re-enable with rate limiting
export PROVIDER_RATE_LIMIT_$PROVIDER=10  # 10 req/sec

# Monitor for 5 minutes
watch -n 1 'curl -s http://localhost:8000/metrics | grep provider_timeout'

# If stable, remove rate limit
unset PROVIDER_RATE_LIMIT_$PROVIDER
unset DISABLED_PROVIDERS
```

## Root Cause Analysis

### Common Causes
1. **Provider Infrastructure Issues**
   - Provider service degradation
   - Network connectivity problems
   - DNS resolution failures

2. **Resource Exhaustion**
   - Memory pressure (check for RESOURCE_EXHAUSTED errors)
   - Thread pool exhaustion
   - Connection pool limits

3. **Configuration Issues**
   - Timeout too aggressive for provider
   - Retry settings causing cascade
   - Circuit breaker thresholds too sensitive

4. **Load Patterns**
   - Sudden traffic spike
   - Large file processing
   - Batch request overload

### Investigation Queries
```bash
# Correlate with other errors
curl http://localhost:8000/metrics | grep -E "ocr_errors_total.*$PROVIDER"

# Check request patterns before timeout spike
grep -B 100 "PROVIDER_TIMEOUT" /app/logs/app.log | \
  grep "request_size\|file_type\|batch_size"

# Memory pressure correlation
sar -r -f /var/log/sa/sa$(date +%d) | \
  grep -A 5 -B 5 "$(date -d '30 minutes ago' +%H:%M)"
```

## Prevention

### Short-term
1. **Monitoring Improvements**
   ```yaml
   # Add predictive alerts
   - alert: ProviderTimeoutTrend
     expr: predict_linear(provider_timeout_rate[10m], 300) > 0.5
     for: 2m
     annotations:
       summary: "Provider {{ $labels.provider }} showing timeout trend"
   ```

2. **Circuit Breaker Tuning**
   ```python
   # Adjust circuit breaker settings
   circuit_breaker_config = {
       "error_threshold": 0.5,  # 50% errors
       "volume_threshold": 10,   # min 10 requests
       "timeout": 60,            # 60 second window
       "recovery_timeout": 30    # 30 second recovery
   }
   ```

### Long-term
1. **Capacity Planning**
   - Analyze provider usage patterns
   - Implement predictive scaling
   - Negotiate SLA improvements

2. **Architecture Improvements**
   - Implement request queuing with backpressure
   - Add provider health scoring
   - Create provider abstraction layer

3. **Timeout Optimization**
   - Dynamic timeout adjustment based on file size
   - Provider-specific timeout profiles
   - Adaptive retry strategies

## Escalation

### Severity Levels
- **Low** (<5 timeouts/min): Monitor, no action
- **Medium** (5-20 timeouts/min): Engineering team notification
- **High** (>20 timeouts/min): Immediate response required
- **Critical** (>50% requests timing out): Incident commander activation

### Contacts
- **Primary On-call**: Check PagerDuty
- **Provider Support**: See provider_contacts.md
- **Engineering Lead**: [Contact via Slack #platform-oncall]

## Related Documentation
- [Provider Configuration Guide](../providers/configuration.md)
- [Circuit Breaker Documentation](../patterns/circuit_breaker.md)
- [Monitoring Guide](../monitoring/guide.md)
- [Provider SLAs](../providers/sla.md)

## Revision History
- 2024-01-20: Initial runbook creation
- 2024-01-20: Added prevention strategies and escalation procedures