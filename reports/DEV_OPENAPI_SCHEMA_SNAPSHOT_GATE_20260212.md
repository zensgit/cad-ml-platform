# DEV_OPENAPI_SCHEMA_SNAPSHOT_GATE_20260212

## Summary

Added an OpenAPI schema snapshot gate to catch unintended contract drift at CI speed,
with a stable snapshot generation workflow and integration into existing OpenAPI checks.

## Changes

- Added contract test: `tests/contract/test_openapi_schema_snapshot.py`
  - Builds a normalized OpenAPI contract snapshot from `/openapi.json`.
  - Compares against baseline file `config/openapi_schema_snapshot.json`.
  - Fails with explicit regenerate command when mismatch occurs.

- Added snapshot generator script: `scripts/ci/generate_openapi_schema_snapshot.py`
  - Generates baseline snapshot JSON.
  - Includes stable normalization for module-prefixed schema names
    (`src__...__ClassName -> ClassName`) to avoid non-semantic naming jitter.

- Added Make target: `openapi-snapshot-update`
  - Updates `config/openapi_schema_snapshot.json`.

- Updated `validate-openapi` in `Makefile`
  - Added snapshot contract test into gate:
    - `tests/contract/test_openapi_operation_ids.py`
    - `tests/contract/test_openapi_schema_snapshot.py`
    - `tests/unit/test_api_route_uniqueness.py`

## Validation

- `make openapi-snapshot-update`
  - Result: generated baseline
  - Evidence: `paths=161`, `operations=166`

- `make validate-openapi`
  - Result: `5 passed`

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `5 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `59 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

OpenAPI drift now has explicit snapshot protection in the fast validation path,
with deterministic baseline update tooling.
