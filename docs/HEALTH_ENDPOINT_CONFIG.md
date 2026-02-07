# Health Endpoint Configuration Visibility

## Overview

The `/health` endpoint has been enhanced to provide comprehensive configuration visibility for operations teams. This allows for better monitoring, debugging, and configuration management without needing access to server logs or configuration files.

## Endpoint Details

**URL**: `GET /health`
**Response**: JSON with system health and configuration

Note: `timestamp` is an ISO 8601, timezone-aware UTC value (e.g., `2025-11-19T10:00:00+00:00`).
Related: `GET /health/extended` includes the same base payload plus vector store and Faiss details.
Classifier cache stats are available at `GET /api/v1/health/classifier/cache` (admin token required).

## Response Structure

```json
{
  "status": "healthy",
  "timestamp": "2025-11-19T10:00:00+00:00",
  "services": {
    "api": "up",
    "ml": "up",
    "redis": "up|disabled"
  },
  "runtime": {
    "python_version": "3.11.0",
    "metrics_enabled": true|false,
    "vision_max_base64_bytes": 1048576,
    "error_rate_ema": {
      "ocr": 0.0,
      "vision": 0.0
    }
  },
  "config": {
    "limits": {
      "vision_max_base64_bytes": 1048576,
      "vision_max_base64_mb": 1.0,
      "ocr_timeout_ms": 30000,
      "ocr_timeout_seconds": 30.0
    },
    "providers": {
      "ocr_default": "auto|paddle|deepseek_hf",
      "confidence_fallback": 0.85
    },
    "monitoring": {
      "error_ema_alpha": 0.2,
      "metrics_enabled": true|false,
      "redis_enabled": true|false,
      "classifier_rate_limit_per_min": 120,
      "classifier_rate_limit_burst": 20,
      "classifier_cache_max_size": 1000
    },
    "network": {
      "cors_origins": ["*"],
      "allowed_hosts": ["*"]
    },
    "debug": {
      "debug_mode": true|false,
      "log_level": "INFO|DEBUG|WARNING|ERROR"
    },
    "ml": {
      "classification": {
        "hybrid_enabled": true|false,
        "hybrid_version": "1.1.0",
        "hybrid_config_path": "config/hybrid_classifier.yaml",
        "graph2d_model_path": "models/graph2d_parts_upsampled_20260122.pth",
        "filename_enabled": true|false,
        "graph2d_enabled": true|false,
        "titleblock_enabled": true|false,
        "process_enabled": true|false
      },
      "sampling": {
        "max_nodes": 200,
        "strategy": "importance|random|hybrid",
        "seed": 42,
        "text_priority_ratio": 0.3
      }
    },
    "core_providers": {
      "bootstrapped": true,
      "bootstrap_timestamp": 1738920000.123,
      "total_domains": 2,
      "total_providers": 4,
      "domains": ["ocr", "vision"],
      "providers": {
        "ocr": ["deepseek_hf", "paddle"],
        "vision": ["deepseek_stub", "stub"]
      }
    }
  }
}
```

## Configuration Sections

### Services Status
- **api**: API service availability
- **ml**: Machine learning models status
- **redis**: Cache service status (disabled if not configured)

### Runtime Information
- **python_version**: Python interpreter version
- **metrics_enabled**: Whether Prometheus metrics are enabled
- **vision_max_base64_bytes**: Current Base64 payload size limit
- **error_rate_ema**: Exponential moving average of error rates for OCR and Vision

### Limits Configuration
- **vision_max_base64_bytes**: Maximum Base64 encoded image size in bytes
- **vision_max_base64_mb**: Same limit in megabytes for easier reading
- **ocr_timeout_ms**: OCR processing timeout in milliseconds
- **ocr_timeout_seconds**: Same timeout in seconds

### Provider Settings
- **ocr_default**: Default OCR provider (auto/paddle/deepseek_hf)
- **confidence_fallback**: Fallback confidence score when not provided

### Monitoring Configuration
- **error_ema_alpha**: Smoothing factor for error rate calculation (0-1)
- **metrics_enabled**: Prometheus metrics export status
- **redis_enabled**: Cache layer status

### Network Settings
- **cors_origins**: Allowed CORS origins
- **allowed_hosts**: Trusted host headers

### Debug Information
- **debug_mode**: Whether debug mode is active
- **log_level**: Current logging level

### ML Configuration
- **classification.hybrid_enabled**: Hybrid classifier master switch
- **classification.*_enabled**: Per-branch switches (filename/graph2d/titleblock/process)
- **classification.hybrid_config_path**: Active runtime config path
- **classification.graph2d_model_path**: Active Graph2D checkpoint path
- **sampling**: Effective DXF graph sampling parameters used in runtime

### Core Provider Registry
- **core_providers.bootstrapped**: Whether startup bootstrap executed
- **core_providers.total_domains**: Number of registered domains
- **core_providers.total_providers**: Total registered provider adapters
- **core_providers.providers**: Domain-to-provider registry mapping

## Provider Registry Endpoints

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/registry
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/providers/registry
```

Returns the runtime registry snapshot used by the new core provider framework.

## Use Cases

### 1. Operations Monitoring
```bash
# Check if size limits are properly configured
curl http://localhost:8000/health | jq '.config.limits'

# Monitor error rates
curl http://localhost:8000/health | jq '.runtime.error_rate_ema'
```

### 2. Troubleshooting
```bash
# Check timeout settings when debugging slow requests
curl http://localhost:8000/health | jq '.config.limits.ocr_timeout_seconds'

# Verify provider configuration
curl http://localhost:8000/health | jq '.config.providers'
```

### 3. Configuration Validation
```bash
# Ensure production settings are correct
curl http://production:8000/health | jq '.config.debug'

# Verify network security settings
curl http://production:8000/health | jq '.config.network'
```

### 4. Automated Monitoring
```python
import requests
import json

def check_config_drift(expected_config):
    """Monitor for configuration drift."""
    response = requests.get("http://localhost:8000/health")
    actual_config = response.json()["config"]

    # Compare limits
    if actual_config["limits"]["vision_max_base64_mb"] != expected_config["vision_max_mb"]:
        alert("Vision size limit has changed!")

    # Check debug mode in production
    if actual_config["debug"]["debug_mode"] == True:
        alert("WARNING: Debug mode enabled in production!")
```

### 5. Classifier Cache Inspection
```bash
curl -H "X-Admin-Token: $ADMIN_TOKEN" http://localhost:8000/api/v1/health/classifier/cache
```
Returns cache size, hit ratio, and hit/miss counts for the classifier API cache.

### 6. Hybrid Config Runtime Inspection
```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/ml/hybrid-config
```
Returns full effective hybrid classifier config (after file/env merge).

### 7. Provider Registry Runtime Inspection
```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/registry
```
Returns bootstrapped status, domain counts, and provider registration map.

## Benefits

1. **Visibility**: Operations teams can see actual running configuration
2. **Debugging**: Easy to verify if configuration changes are applied
3. **Monitoring**: Can track configuration drift and compliance
4. **Documentation**: Self-documenting configuration through API
5. **Automation**: Easy to build monitoring and alerting around config

## Security Considerations

- The `/health` endpoint exposes configuration but no sensitive data
- Consider restricting access in production if configuration is sensitive
- No passwords, keys, or secrets are exposed
- Network settings (CORS, hosts) are visible for security auditing

## Integration with Monitoring

### Prometheus Metrics
```yaml
# prometheus.yml
- job_name: 'cad_ml_health'
  metrics_path: '/health'
  static_configs:
    - targets: ['localhost:8000']
```

### Grafana Dashboard
Create alerts based on configuration values:
- Alert if `vision_max_base64_mb` < 1.0
- Alert if `error_rate_ema` > 0.1
- Alert if `debug_mode` = true in production

## Maintenance

The health endpoint configuration is automatically updated when settings change. No manual maintenance required.

---

*Last Updated: 2026-02-07*
*Implementation: [`src/api/health_utils.py`](../src/api/health_utils.py), [`src/api/v1/health.py`](../src/api/v1/health.py), [`src/core/providers/bootstrap.py`](../src/core/providers/bootstrap.py), [`src/main.py`](../src/main.py)*
