# Compare Endpoint Metrics (2025-12-31)

## Scope

- Emit request metrics for `/api/compare` in cad-ml-platform.

## Changes

- `src/utils/analysis_metrics.py`: add `compare_requests_total` counter.
- `src/api/v1/compare.py`: increment counter on success and error paths.

## Tests

```bash
.venv/bin/python -m pytest tests/unit/test_compare_endpoint.py -v
```

Result:
- `3 passed in 4.53s`
