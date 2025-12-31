# DedupCAD Vision Integration Final Summary (2026-01-01)

## Completed Work

- Audited cad-ml ↔ dedupcad-vision integration paths and documented the contract.
- Verified feature vector ordering and compare/hash expectations.
- Added payload format + legacy fallback metrics, plus optional dependency guards.
- Added retry/timeout/circuit-breaker unit tests for the vision client.
- Added async precision overlay integration test coverage.
- Updated staging + production runbooks with circuit and rollout metrics.

## Tests (this phase)

- `pytest tests/unit/test_dedup_2d_jobs_redis.py -v` (23 passed, 3 skipped)
- `pytest tests/unit/test_dedupcad_vision_client.py -v` (3 passed)
- `pytest tests/test_dedup_2d_proxy.py -k "async_with_precision_overlay" -v` (1 passed, 28 deselected)

## Notes

- Metrics counter assertions are skipped when `prometheus_client` is unavailable.
- Missing optional dependencies (ARQ/PyJWT) now fail gracefully with explicit errors when required.
