# DEV_TENANT_CONTEXT_CONTEXTVARS_LEAK_FIX_20260213

## Summary

Fixed tenant context leakage across tests by replacing a shared class-level mutable field in
`TenantContext` with `contextvars.ContextVar` token-based enter/exit restoration.

## Problem

- `TenantContext` in `src/core/assistant/multi_tenant.py` stored active tenant in a class variable.
- Under concurrent execution (stress tests + thread pools), context state could be overwritten and
  leak into subsequent tests, causing:
  - `tests/unit/assistant/test_multi_tenant.py::TestTenantContext::test_context_entry_exit`
  - `tests/unit/assistant/test_multi_tenant.py::TestTenantContext::test_get_current_without_context`

## Changes

- `src/core/assistant/multi_tenant.py`
  - Added `ContextVar` + `Token` import.
  - Replaced `_current_tenant: Optional[Tenant]` with:
    - `_current_tenant: ContextVar[Optional[Tenant]] = ContextVar(..., default=None)`
  - Updated context manager:
    - `__enter__` now calls `.set()` and stores token.
    - `__exit__` now calls `.reset(token)` and clears token.
  - `get_current()` now returns `ContextVar.get()`.

- `tests/unit/assistant/test_multi_tenant.py`
  - Added regression test `test_context_isolation_across_threads` to ensure tenant context does
    not leak between threads and remains correct in nested concurrent scenarios.

## Validation

Executed:

```bash
.venv/bin/python -m pytest tests/unit/assistant/test_multi_tenant.py -k "context_entry_exit or get_current_without_context or context_isolation_across_threads" -v
.venv/bin/python -m pytest tests/stress/test_load_simulation.py -k "concurrent_quota_operations or tenant_context_cleanup" -v
make validate-core-fast
```

Results:

- `tests/unit/assistant/test_multi_tenant.py` targeted set: `3 passed`
- `tests/stress/test_load_simulation.py` targeted set: `2 passed`
- `make validate-core-fast`: `passed` (tolerance, openapi, service-mesh, provider core/contract)

## Risk Assessment

- Low risk:
  - Behavior is now context-local and concurrency-safe.
  - Nested context restoration semantics preserved via token reset.
  - Added unit regression test for thread isolation.

