# DEV_DASHBOARD_METRICS_ALIGNMENT_SECURITY_ROLLBACK_V4_VALIDATION_20260106

## Scope
Validate dashboard metric references and new security/rollback/v4 metrics wiring.

## Commands
- `python3 scripts/validate_dashboard_metrics.py`
- `pytest tests/unit/test_model_rollback_health.py tests/unit/test_model_rollback_level3.py tests/unit/test_model_opcode_modes.py tests/unit/test_v4_feature_performance.py -v`

## Results
- Dashboard validator: success.
- Tests: 59 passed, 4 skipped (`prometheus client disabled in this environment`).
