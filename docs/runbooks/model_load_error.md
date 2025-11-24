# Runbook: MODEL_LOAD_ERROR

## Overview
This runbook addresses errors that occur when OCR/Vision models fail to load or initialize properly. These errors can prevent providers from starting or cause runtime failures.

## Error Code
- **Code**: `MODEL_LOAD_ERROR`
- **Severity**: Critical
- **Impact**: Provider unavailable, service degradation, potential cascade failures

## Detection

### Metrics
```promql
# Model load failures
sum(rate(ocr_errors_total{code="MODEL_LOAD_ERROR"}[5m])) > 0

# Provider initialization failures
ocr_provider_initialization_failed_total > 0

# Recording rule
model_load_failure_rate{provider="$provider"} > 0
```

### Logs
```bash
# Check for model loading errors
grep -E "MODEL_LOAD_ERROR|Failed to load model|Model initialization failed" \
  /app/logs/app.log | tail -50

# Provider-specific initialization
journalctl -u cad-ml-platform --since "10 minutes ago" | \
  grep -E "provider.*init|load.*model|initialize.*failed"

# Memory errors during load
dmesg | grep -i "out of memory"
grep "MemoryError" /app/logs/*.log
```

### Dashboard
- Grafana Dashboard: "CAD ML Platform - Observability"
- Panel: "Provider Health Score"
- Look for providers with health score = 0 (indicates initialization failure)

## Response Steps

### 1. Immediate Actions (0-5 minutes)

#### Identify affected providers
```bash
# List all provider statuses
curl http://localhost:8000/health | jq '.providers | to_entries[] |
  select(.value.status != "healthy") | {provider: .key, status: .value}'

# Check which models failed
curl http://localhost:8000/api/v1/ocr/providers | jq '.[] |
  select(.model_loaded == false)'

# Review startup logs
journalctl -u cad-ml-platform -n 100 --no-pager | \
  grep -A 5 -B 5 "MODEL_LOAD_ERROR"
```

#### Prevent cascade failures
```bash
# Disable affected provider to prevent repeated attempts
export DISABLED_PROVIDERS="$PROVIDER"

# Or mark provider as unavailable in config
echo "providers.$PROVIDER.enabled: false" >> /app/config/overrides.yaml
systemctl reload cad-ml-platform
```

### 2. Investigation (5-15 minutes)

#### Check model files
```bash
# Verify model files exist
ls -la /app/models/$PROVIDER/
find /app/models -name "*.bin" -o -name "*.onnx" -o -name "*.pt" | \
  xargs ls -lh

# Check file integrity
md5sum /app/models/$PROVIDER/* | \
  diff - /app/models/$PROVIDER/checksums.txt

# Verify permissions
ls -la /app/models/$PROVIDER/ | grep -v "rw-r--r--"
```

#### Analyze resource constraints
```bash
# Memory availability
free -h
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|SwapTotal"

# Disk space for model storage
df -h /app/models
du -sh /app/models/*

# GPU availability (if applicable)
nvidia-smi
lspci | grep -i nvidia
```

#### Test model loading directly
```python
# Test model loading in isolation
import sys
import traceback
from src.core.ocr.providers import get_provider_class

provider_class = get_provider_class("$PROVIDER")
try:
    provider = provider_class()
    provider._load_model()
    print(f"Model loaded successfully for {provider.__class__.__name__}")
except Exception as e:
    print(f"Model load failed: {e}")
    traceback.print_exc()
```

### 3. Common Fixes (15-30 minutes)

#### Fix: Insufficient Memory
```bash
# Increase container memory limits
kubectl set resources deployment cad-ml-platform \
  --limits=memory=8Gi --requests=memory=4Gi

# Or adjust Docker limits
docker update --memory="8g" --memory-swap="8g" cad-ml-platform

# Enable swap if needed
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Fix: Missing Model Files
```bash
# Re-download models
cd /app
python scripts/download_models.py --provider $PROVIDER

# Or restore from backup
aws s3 cp s3://ml-models-backup/$PROVIDER/ /app/models/$PROVIDER/ --recursive

# Verify download
md5sum /app/models/$PROVIDER/*.bin
```

#### Fix: Corrupted Model Files
```bash
# Clear model cache
rm -rf /app/models/$PROVIDER/.cache
rm -rf /tmp/transformers_cache

# Re-download with verification
curl -O https://models.example.com/$PROVIDER/model.bin
echo "expected_checksum model.bin" | md5sum -c
mv model.bin /app/models/$PROVIDER/
```

#### Fix: Permission Issues
```bash
# Fix ownership
chown -R app:app /app/models

# Fix permissions
find /app/models -type f -exec chmod 644 {} \;
find /app/models -type d -exec chmod 755 {} \;

# SELinux context (if applicable)
restorecon -Rv /app/models
```

### 4. Recovery (30-60 minutes)

#### Restart with monitoring
```bash
# Clear any error state
redis-cli DEL "provider:$PROVIDER:error_state"

# Restart service with debug logging
export LOG_LEVEL=DEBUG
systemctl restart cad-ml-platform

# Monitor logs during startup
journalctl -u cad-ml-platform -f | grep -E "$PROVIDER|model|load"
```

#### Validate model loading
```bash
# Test provider initialization
curl -X POST http://localhost:8000/api/v1/ocr/test \
  -H "Content-Type: application/json" \
  -d '{"provider": "'$PROVIDER'", "init_only": true}'

# Verify model in memory
ps aux | grep python | head -1  # Get PID
pmap -x $PID | grep -i model
```

#### Re-enable provider
```bash
# Remove from disabled list
unset DISABLED_PROVIDERS

# Update configuration
sed -i "/$PROVIDER.*enabled.*false/d" /app/config/overrides.yaml

# Gradual re-introduction
export PROVIDER_RATE_LIMIT_$PROVIDER=1  # Start with 1 req/sec
# Monitor for 5 minutes
sleep 300
export PROVIDER_RATE_LIMIT_$PROVIDER=10  # Increase if stable
```

## Root Cause Analysis

### Common Causes

1. **Resource Constraints**
   - Insufficient memory for model loading
   - Disk space exhaustion
   - GPU memory limitations
   - Container resource limits

2. **Model File Issues**
   - Missing model files
   - Corrupted downloads
   - Version mismatches
   - Incorrect file permissions

3. **Dependencies**
   - Missing Python packages (transformers, torch, onnxruntime)
   - Version conflicts
   - Native library issues (CUDA, cuDNN)

4. **Configuration Problems**
   - Incorrect model paths
   - Wrong model architecture specified
   - Invalid model parameters
   - Environment variable issues

### Deep Investigation
```bash
# Check for OOM killer activity
dmesg | grep -i "killed process"
journalctl -xe | grep -i "oom"

# Review dependency versions
pip list | grep -E "transformers|torch|onnx"
ldd /usr/local/lib/python*/site-packages/torch/lib/*.so | grep "not found"

# Analyze provider initialization timeline
grep -h "provider.*init" /app/logs/*.log | \
  awk '{print $1, $2, $NF}' | sort

# Memory usage during model load
python -c "
import tracemalloc
tracemalloc.start()
# Load model
from src.core.ocr.providers.$PROVIDER import Provider
p = Provider()
current, peak = tracemalloc.get_traced_memory()
print(f'Peak memory: {peak / 10**6:.1f} MB')
"
```

## Prevention

### Short-term
1. **Pre-flight Checks**
   ```python
   # Add to provider initialization
   def pre_load_checks(self):
       # Check available memory
       import psutil
       if psutil.virtual_memory().available < self.min_memory_required:
           raise ResourceError("Insufficient memory")

       # Verify model files
       if not os.path.exists(self.model_path):
           raise FileNotFoundError(f"Model not found: {self.model_path}")

       # Check file integrity
       if not self.verify_checksum():
           raise ValueError("Model file corrupted")
   ```

2. **Graceful Degradation**
   ```yaml
   # Enable stub models for failures
   providers:
     $PROVIDER:
       fallback_mode: stub
       stub_confidence: 0.0
       stub_response: "Model unavailable"
   ```

### Long-term
1. **Model Management System**
   - Centralized model registry
   - Automatic model versioning
   - Health checks before deployment
   - A/B testing infrastructure

2. **Resource Planning**
   - Model memory profiling
   - Dynamic resource allocation
   - Lazy loading strategies
   - Model quantization/optimization

3. **Monitoring Enhancements**
   ```yaml
   # Add proactive alerts
   - alert: ModelMemoryPressure
     expr: |
       (node_memory_MemAvailable_bytes / 1024^3) <
       (sum(ocr_model_size_bytes{state="loaded"}) * 1.5 / 1024^3)
     annotations:
       summary: "Insufficient memory for model loading"

   - alert: ModelFilesMissing
     expr: ocr_model_files_present == 0
     for: 1m
     annotations:
       summary: "Model files missing for {{ $labels.provider }}"
   ```

## Escalation

### Severity Matrix
| Condition | Severity | Response Time | Escalation |
|-----------|----------|---------------|------------|
| Single provider affected | Medium | 30 min | Engineering on-call |
| Multiple providers affected | High | 15 min | Engineering + Platform |
| All providers down | Critical | 5 min | Incident Commander |
| Data corruption suspected | Critical | Immediate | Engineering + Security |

### Communication Template
```
Subject: MODEL_LOAD_ERROR - $PROVIDER

Status: [Investigating|Mitigating|Resolved]
Impact: $PROVIDER unavailable, X% traffic affected
Started: [timestamp]
ETA: [estimate]

Current Actions:
- [Action 1]
- [Action 2]

Next Steps:
- [Step 1]
- [Step 2]

Updates: Every 30 minutes or on status change
```

## Related Documentation
- [Model Management Guide](../models/management.md)
- [Provider Configuration](../providers/configuration.md)
- [Resource Requirements](../deployment/resources.md)
- [Dependency Management](../development/dependencies.md)

## Revision History
- 2024-01-20: Initial runbook creation
- 2024-01-20: Added prevention strategies and resource planning