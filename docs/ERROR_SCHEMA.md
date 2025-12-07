# Error Schema Documentation

All API errors follow a unified structured format.

## Schema

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "stage": "processing_stage",
    "message": "Human readable message",
    "severity": "error|warning|info",
    "context": {
      "key": "value"
    },
    "suggestion": "Optional suggestion"
  }
}
```

## Common Error Codes

| Code | Description |
|------|-------------|
| `INTERNAL_ERROR` | Unexpected server error |
| `INPUT_VALIDATION_FAILED` | Invalid input parameters |
| `DATA_NOT_FOUND` | Resource not found |
| `DIMENSION_MISMATCH` | Vector dimension mismatch |
| `MODEL_SECURITY_VIOLATION` | Model security check failed |
| `SERVICE_UNAVAILABLE` | Dependency service down |

## Examples

### Model Security Violation
```json
{
  "detail": {
    "code": "MODEL_SECURITY_VIOLATION",
    "stage": "model_reload",
    "message": "Blocked opcode detected",
    "context": {
      "opcode": "GLOBAL"
    }
  }
}
```

### Migration Preview Error
```json
{
  "detail": {
    "code": "INPUT_VALIDATION_FAILED",
    "stage": "migration_preview",
    "message": "Unsupported target feature version",
    "context": {
      "to_version": "v5",
      "allowed": ["v1", "v2", "v3", "v4"]
    }
  }
}
```
