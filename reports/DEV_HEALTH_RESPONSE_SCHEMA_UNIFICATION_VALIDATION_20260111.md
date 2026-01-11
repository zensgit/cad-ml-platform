# DEV_HEALTH_RESPONSE_SCHEMA_UNIFICATION_VALIDATION_20260111

## Scope
Validate the unified health response schemas for `/health` and `/health/extended`.

## Commands
- `python3 -m pytest tests/unit/test_main_coverage.py -k health_check -v`
- `python3 -m pytest tests/unit/test_main_coverage.py -k "readiness or extended_health" -v`
- `python3 -m pytest tests/unit/test_health_extended_endpoint.py -v`

## Results
- Health endpoint unit coverage and extended endpoint tests passed.

## Notes
- `/health/extended` now returns the base health payload alongside vector/Faiss details.
