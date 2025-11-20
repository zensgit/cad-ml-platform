# Health Endpoint Configuration Visibility

## Overview

The `/health` endpoint has been enhanced to provide comprehensive configuration visibility for operations teams. This allows for better monitoring, debugging, and configuration management without needing access to server logs or configuration files.

## Endpoint Details

**URL**: `GET /health`
**Response**: JSON with system health and configuration

Note: `timestamp` is an ISO 8601, timezone-aware UTC value (e.g., `2025-11-19T10:00:00+00:00`).

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
      "redis_enabled": true|false
    },
    "network": {
      "cors_origins": ["*"],
      "allowed_hosts": ["*"]
    },
    "debug": {
      "debug_mode": true|false,
      "log_level": "INFO|DEBUG|WARNING|ERROR"
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

*Last Updated: 2025-11-19*
*Implementation: [`src/main.py:99-150`](../src/main.py#L99)*
