# DEV_PROVIDER_INSTANCE_CACHE_AND_HEALTH_UNIFICATION_20260208

## Goal
1. Add **provider instance caching** so repeated calls to `ProviderRegistry.get(domain, name)`
   can reuse a singleton instance (reduced overhead; stable status snapshots).
2. Unify **health/readiness semantics** so `/ready` core provider checks and
   `/api/v1/providers/health` use the same timeout-bounded logic and update provider
   runtime status consistently.

## Changes
### Provider instance caching
- `src/core/providers/registry.py`
  - Added cached singleton instances for the default `get(domain, name)` path
    (no args/kwargs).
  - Controlled by `PROVIDER_REGISTRY_CACHE_ENABLED` (default: `true`).
  - Added `ProviderRegistry.clear_instances()` to clear cached instances without
    removing provider registrations.

- `tests/conftest.py`
  - Added an autouse fixture to clear cached provider instances between tests to
    prevent cross-test interference (tests often monkeypatch env/config/factories).

### Health/readiness unification
- `src/core/providers/base.py`
  - Extended `BaseProvider.health_check(timeout_seconds=...)`:
    - timeout-bounded via `asyncio.wait_for`
    - sets `last_error="timeout"` on timeout
    - sets `last_error="unhealthy"` when `_health_check_impl()` returns `False`

- `src/core/providers/readiness.py`
  - Readiness checks now call `provider.health_check(timeout_seconds=...)` so provider
    status snapshots are updated the same way as the provider health endpoint.

- `src/api/v1/health.py`
  - `/api/v1/providers/health` now uses `provider.health_check(timeout_seconds=...)`
    instead of calling private `_health_check_impl()` and mutating private fields.

## Verification
Commands:
```bash
python3 -m pytest -q tests/unit/test_provider_framework.py
python3 -m pytest -q tests/unit/test_provider_readiness.py
python3 -m pytest -q tests/unit/test_provider_health_endpoint.py
python3 -m pytest -q tests/unit/test_provider_framework_classifier_bridge.py
```

Result:
- All tests passed locally.

