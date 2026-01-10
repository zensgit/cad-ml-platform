# DEV_DEDUP2D_STORAGE_METRICS_TEST_20260101

## Scope
- Run targeted unit test for dedup2d storage metrics.

## Command
```bash
pytest tests/unit/test_dedup2d_file_storage_metrics.py -v
```

## Result
- `1 skipped` (prometheus_client unavailable in local runtime).

## Follow-up
- Run metrics contract test in full environment:
  `pytest tests/test_metrics_contract.py -k dedup2d_storage_metrics_exposed -v`
