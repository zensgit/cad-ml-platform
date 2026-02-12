# DEV_PROVIDER_FRAMEWORK_HARDENING_20260212

## Summary

Hardened the core provider framework to fail fast on invalid provider
registrations and to keep health/readiness checks backward-compatible with
legacy provider `health_check()` signatures.

## Changes

- Updated `src/core/providers/registry.py`
  - Added registration token validation for `domain` / `provider_name`.
  - Rejected reserved separators (`/`, `:`) in provider IDs.
  - Added runtime class contract check: registered classes must inherit
    `BaseProvider`.

- Updated `src/core/providers/readiness.py`
  - Added compatibility invocation path for providers that only implement
    `health_check()` without `timeout_seconds`.
  - Preserved existing timeout-bounded behavior for modern providers.

- Updated `src/api/v1/health.py`
  - Added compatibility invocation path for legacy `health_check()` signatures
    in `/api/v1/providers/health`.
  - Added snapshot fallback builder when provider objects do not expose
    `status_snapshot()`.

- Added tests
  - `tests/unit/test_registry_coverage.py`
    - registration rejects non-`BaseProvider` classes
    - registration rejects empty/invalid provider IDs
  - `tests/unit/test_provider_health_endpoint.py`
    - provider health endpoint supports legacy provider signatures
  - `tests/unit/test_readiness_coverage.py`
    - readiness supports legacy provider signatures

## Validation

- `.venv/bin/python -m pytest tests/unit/test_registry_coverage.py tests/unit/test_provider_health_endpoint.py tests/unit/test_readiness_coverage.py tests/unit/test_provider_readiness.py -v`
  - Result: `55 passed`

- `make validate-core-fast`
  - Result: passed
  - Evidence:
    - tolerance suite: `48 passed`
    - openapi/route suite: `5 passed`
    - service-mesh suite: `103 passed`
    - provider-core suite: `60 passed`
    - provider-contract suite: `4 passed, 20 deselected`

## Outcome

Provider extensibility is safer (invalid classes blocked at registration) and
runtime probes are more resilient to legacy plugins without sacrificing current
contract checks.
