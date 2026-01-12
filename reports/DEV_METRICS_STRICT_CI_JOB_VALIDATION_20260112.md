# DEV_METRICS_STRICT_CI_JOB_VALIDATION_20260112

## Change
- Added `metrics-contract-strict` job in `.github/workflows/observability-checks.yml`.

## Local Verification
- STRICT_METRICS=1 pytest tests/test_metrics_contract.py -v

## Result
- 21 passed, 1 skipped in 2.21s

## CI Notes
- The new job runs the same command in CI; run history will be captured by Actions.
