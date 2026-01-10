# DEV_METRICS_PR_SUMMARY_20260106

## Overview
This update expands observability coverage across cache tuning, model security/rollback, v4 feature metrics, vector migration, and dashboard alignment. It adds new counters/gauges/histograms, aligns Grafana queries with exported metrics, and verifies contracts + unit coverage under metrics-enabled environments.

## Key Changes
- Added cache tuning POST endpoint metrics and recommendation gauges.
- Added model opcode mode gauge and opcode scan/blocked counters.
- Added model rollback counters and gauges for rollback level and snapshots available.
- Added v4 surface count and shape entropy histograms (with _count assertions).
- Added model interface validation failure counter during reload.
- Added vector migrate downgrade counters and dimension-delta histogram.
- Aligned Grafana dashboard panel queries with exported metric names and drift histogram quantiles.
- Added GitHub Actions Docker staging smoke workflow + local compose validation script.

## Validation
- `.venv/bin/python -m pytest tests/test_metrics_contract.py -v` (19 passed, 3 skipped)
- `.venv/bin/python -m pytest tests/unit/test_cache_tuning.py tests/unit/test_model_opcode_modes.py tests/unit/test_v4_feature_performance.py -v` (43 passed)
- `.venv/bin/python -m pytest tests/unit/test_model_security_validation.py tests/unit/test_model_rollback_health.py tests/unit/test_model_rollback_level3.py tests/unit/test_vector_migrate_metrics.py tests/unit/test_vector_migrate_dimension_histogram.py -v` (61 passed)
- `.venv/bin/python -m pytest tests/unit -k metrics -v` (223 passed, 3500 deselected)
- `python3 scripts/validate_dashboard_metrics.py` (pass)
- `bash scripts/ci/docker_staging_smoke.sh` (local build blocked by Docker Hub pull; CI run failed on filelock pin; see staging report)

## Notes / Risks
- Metrics-dependent tests may skip under system Python if `prometheus_client` is unavailable; use `.venv` runs above for full coverage.
- No functional regressions observed in metrics-enabled test runs.
