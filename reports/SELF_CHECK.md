# Self-Check Report

## Scope
- Run enhanced self-check in strict metrics mode with counter increments.

## Changes
- scripts/self_check.py: ensure project root is on `sys.path`.
- src/core/vision/manager.py: use ErrorCode values for metrics and handle provider errors as external service errors.
- src/api/v1/ocr.py: use ErrorCode values for OCR error metrics and align stages to allowed set.
- src/api/v1/analyze.py: use ErrorCode.INTERNAL_ERROR for analysis_errors_total code.
- tests/test_metrics_consistency.py: update vision_errors_total label expectations to ErrorCode.

## Test Run
- Command: `SELF_CHECK_STRICT_METRICS=1 SELF_CHECK_INCREMENT_COUNTERS=1 .venv/bin/python scripts/self_check.py`
- Result: `All checks passed`

## Notes
- Vision self-check attempted Anthropic provider and received 401 (invalid key); metrics now label this as EXTERNAL_SERVICE_ERROR per ErrorCode.
