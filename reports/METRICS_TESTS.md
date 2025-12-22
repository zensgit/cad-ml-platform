# Metrics Tests Report

## Scope
- Align OCR metric label test values with allowed ErrorCode/stage lists.
- Avoid unawaited coroutine warning in DeepSeekHfProvider timeout test.

## Changes
- tests/unit/test_metrics_coverage.py: use ErrorCode.INTERNAL_ERROR + stage=infer for label coverage.
- tests/test_metrics_consistency.py: patch asyncio.wait_for with a coroutine-safe timeout helper.

## Test Run
- Command: `.venv/bin/python -m pytest tests/test_health_and_metrics.py tests/test_metrics_contract.py tests/test_metrics_consistency.py tests/unit/test_metrics_export_presence.py tests/unit/test_metrics_coverage.py tests/unit/test_metrics_helpers_coverage.py -q`
- Result: `102 passed, 2 skipped in 44.07s`

## Notes
- Tests executed with repo-local virtualenv at `.venv` (Python 3.11.13).
