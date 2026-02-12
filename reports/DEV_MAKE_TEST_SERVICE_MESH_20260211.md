# DEV_MAKE_TEST_SERVICE_MESH_20260211

## Summary
- Added a dedicated Make target for service-mesh regression tests to reduce manual command drift.

## Changes
- Updated `Makefile`
  - Added `.PHONY` entry: `test-service-mesh`
  - Added target: `make test-service-mesh`
  - Included suites:
    - `tests/unit/test_load_balancer_coverage.py`
    - `tests/unit/test_service_discovery_coverage.py`

## Validation
- `make test-service-mesh`
  - Result: `103 passed`

