# DEV_DEDUP2D_CHECKLIST_TESTS_20251224

## Scope
- Run Dedup2D PR checklist tests from cad-ml-platform.

## Validation
- Command: `.venv/bin/python -m pytest -q tests/unit/test_dedup2d_job_list_redis.py`
  - Result: `2 passed`.
- Command: `.venv/bin/python -m pytest -q tests/test_dedup_2d_proxy.py`
  - Result: `27 passed`.
- Command: `.venv/bin/python -m pytest -q tests/unit/test_dedup2d_webhook.py`
  - Result: `4 passed`.
- Command: `.venv/bin/python -m pytest -q tests/test_metrics_contract.py`
  - Result: `18 passed, 2 skipped`.

## Notes
- Skips in metrics contract are expected (optional exporters).
