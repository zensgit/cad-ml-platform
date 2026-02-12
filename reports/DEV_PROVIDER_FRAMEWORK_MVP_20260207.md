# DEV_PROVIDER_FRAMEWORK_MVP_20260207

## Summary

Implemented a minimal, reusable provider framework under `src/core/providers`.

## Why This Was Needed

- Existing project already has domain-level provider abstractions (`vision` / `ocr`), but a shared cross-domain provider framework was incomplete.
- `src/core/providers/__init__.py` existed, but imports failed because `base.py` and `registry.py` were missing.
- This patch converts that incomplete shell into a working MVP without changing existing vision/ocr runtime paths.

## Changes

### 1. Core Provider Base

Added `src/core/providers/base.py`:

- `ProviderStatus`: `unknown|healthy|degraded|down`
- `ProviderConfig`: generic config dataclass (`name`, `provider_type`, `enabled`, `timeout_seconds`, `metadata`)
- `BaseProvider[ConfigT, ResultT]`:
  - async `process()` entrypoint
  - async `health_check()` with status/error/latency updates
  - optional lifecycle hooks: `warmup()`, `shutdown()`
  - helpers: `mark_degraded()`, `mark_healthy()`, `status_snapshot()`

### 2. Provider Registry

Added `src/core/providers/registry.py`:

- Decorator registration: `@ProviderRegistry.register(domain, provider_name)`
- Instance creation: `ProviderRegistry.get(domain, provider_name, *args, **kwargs)`
- Class lookup: `get_provider_class()`
- Discovery: `list_domains()`, `list_providers(domain)`
- Safety and maintenance: `exists()`, `unregister()`, `clear()`
- Thread-safety via `RLock`

### 3. Unit Tests

Added `tests/unit/test_provider_framework.py`:

- register/list/exists flow
- duplicate registration rejection
- unknown provider lookup failure
- provider instantiation and async process call
- health check success snapshot
- health check exception path (`status=down`, `last_error` captured)
- unregister behavior

## Validation

Executed:

```bash
python3 - <<'PY'
import importlib
m = importlib.import_module('src.core.providers')
print('import_ok', sorted(getattr(m, '__all__', [])))
PY

pytest tests/unit/test_provider_framework.py -q
python3 -m black src/core/providers tests/unit/test_provider_framework.py --check
python3 -m flake8 src/core/providers tests/unit/test_provider_framework.py --max-line-length=100
```

Results:

- provider package import: PASS
- unit tests: PASS (`7 passed`)
- formatting check: PASS
- lint check: PASS

## Compatibility Notes

- No changes were made to existing `vision` and `ocr` provider pipelines.
- This is an additive MVP foundation for future unification/adapters.
