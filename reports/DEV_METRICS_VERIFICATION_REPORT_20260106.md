# DEV_METRICS_VERIFICATION_REPORT_20260106

## Scope
Capture validation results for the 2026-01-06 metrics workstream.

## Metrics-Enabled Validation (.venv)
- `.venv/bin/python -m pytest tests/test_metrics_contract.py -v` (19 passed, 3 skipped).
- `.venv/bin/python -m pytest tests/unit/test_cache_tuning.py tests/unit/test_model_opcode_modes.py tests/unit/test_v4_feature_performance.py -v` (43 passed).
- `.venv/bin/python -m pytest tests/unit/test_model_security_validation.py tests/unit/test_model_rollback_health.py tests/unit/test_model_rollback_level3.py tests/unit/test_vector_migrate_metrics.py tests/unit/test_vector_migrate_dimension_histogram.py -v` (61 passed).
- `.venv/bin/python -m pytest tests/unit -k metrics -v` (223 passed, 3500 deselected).
- `python3 scripts/validate_dashboard_metrics.py` (pass).

## Metrics-Disabled Validation (system Python)
- `pytest tests/test_metrics_contract.py -k metric_label_schemas -v` (skipped: metrics disabled).
- `pytest tests/unit/test_cache_tuning.py tests/unit/test_model_opcode_modes.py tests/unit/test_v4_feature_performance.py -v` (39 passed, 4 skipped).

## Docker Staging Smoke
- `bash scripts/ci/docker_staging_smoke.sh` (build blocked while pulling `python:3.9-slim`; aborted).
- `SKIP_BUILD=1 bash scripts/ci/docker_staging_smoke.sh` (POST cache tuning returned 405 with stale image).
- GitHub Actions `docker-staging-smoke.yml` failed on `filelock==3.20.1` during Docker build (fixed by pinning `filelock==3.19.1`).
- Re-run failed on `urllib3==2.6.0` conflicting with `botocore` constraints; pinned to `urllib3==2.1.0`.
- Re-run failed on `urllib3==2.1.0`; pinned to `urllib3==2.0.7`.
- Re-run failed on `urllib3==2.0.7`; pinned to `urllib3==1.26.20` and re-run pending.
- Re-run failed on `pythonocc-core>=7.7.0` (no wheel for Python 3.9); update Docker base to `python:3.10-slim` and re-run pending.
- Report: `reports/DEV_GITHUB_DOCKER_STAGING_WORKFLOW_VALIDATION_20260110.md`

## Notes
- Use `.venv` results above for full metrics coverage when `prometheus_client` is available.
