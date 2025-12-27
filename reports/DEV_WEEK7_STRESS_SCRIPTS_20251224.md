# DEV_WEEK7_STRESS_SCRIPTS_20251224

## Scope
- Run stress scripts for concurrency reload and degradation flapping.
- Fix flapping script health endpoint path.

## Changes
- `scripts/stress_degradation_flapping.py`
  - Update health endpoint to `/api/v1/health/faiss/health`.

## Validation
- Command: `python3 scripts/stress_concurrency_reload.py --threads 2 --iterations 2`
  - Result: HTTP 403 for all requests (unauthorized with default keys).
- Command: `python3 scripts/stress_concurrency_reload.py --threads 2 --iterations 2 --api-key test --admin-token test`
  - Result: HTTP 500 for all requests (server error during reload).
- Command: `python3 scripts/stress_degradation_flapping.py --cycles 3 --interval 0.1 --api-key test`
  - Result: PASS (3/3 OK, no flapping).

## Notes
- Reload stress results depend on valid admin credentials and server configuration.
- Flapping script now targets the active Faiss health endpoint used by integration tests.
