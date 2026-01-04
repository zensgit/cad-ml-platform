# DEV_DEDUP2D_METRICS_CONTRACT_TEST_20260101

## Scope
- Validate dedup2d storage metrics presence in metrics contract test.

## Command
```bash
pytest tests/test_metrics_contract.py -k dedup2d_storage_metrics_exposed -v
```

## Result
- `1 skipped` because metrics client is disabled in this runtime (`/metrics` returned `app_metrics_disabled`).

## Fix Applied
- Updated test to skip when metrics are disabled to avoid false negatives in minimal envs.

## Follow-up
- Run the same test in a full environment with `prometheus_client` installed to assert metric exposure.

## Update
### Command
```bash
pytest tests/test_metrics_contract.py -k metrics -v
```

### Result
- `8 passed, 14 skipped` (metrics disabled; fallback metrics check passed).

### Notes
- Metrics contract tests now gate on `/health` `metrics_enabled` (with `/metrics` fallback).

## Update (metrics enabled)
### Command
```bash
source .venv-metrics/bin/activate
python -m pytest tests/test_metrics_contract.py -k metrics -v
```

### Result
- `19 passed, 3 skipped` (fallback test skipped; strict mode tests skipped).

### Notes
- Metrics enabled via `prometheus_client` installed in `.venv-metrics`.

## Update (strict metrics)
### Command
```bash
source .venv-metrics/bin/activate
STRICT_METRICS=1 python -m pytest tests/test_metrics_contract.py -k strict -v
```

### Result
- `2 passed, 20 deselected`

### Notes
- Strict checks now trigger error metrics on demand and verify active provider coverage.
