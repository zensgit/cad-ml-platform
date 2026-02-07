# DEV_PROVIDER_FRAMEWORK_OCR_BOOTSTRAP_HEALTH_20260207

## Summary

Completed the next provider-framework phase requested as `1+2+3`:

1. Added OCR bridge for the core provider framework.
2. Added automatic core provider registry bootstrap at app startup.
3. Added provider registry visibility in health payload and dedicated health API endpoints.

## Scope

### 1) OCR Bridge

Added `src/core/providers/ocr.py`:

- `OcrProviderConfig` (extends `ProviderConfig`)
- `OcrProviderAdapter` (extends `BaseProvider`)
  - Validates raw bytes input
  - Delegates extraction to wrapped `OcrClient`
  - Supports `trace_id` default/override
  - Health check delegates to wrapped provider when available
- `bootstrap_core_ocr_providers()`
  - Registers `ocr/paddle`
  - Registers `ocr/deepseek_hf`

### 2) Startup Bootstrap

Added `src/core/providers/bootstrap.py`:

- `bootstrap_core_provider_registry()`
  - Bootstraps both vision and OCR domains
  - Marks bootstrap state/timestamp
- `get_core_provider_registry_snapshot(lazy_bootstrap=True)`
  - Returns serializable runtime registry snapshot
  - Supports lazy bootstrap when snapshot requested before app lifespan startup

Updated startup lifecycle in `src/main.py`:

- Calls `bootstrap_core_provider_registry()` during app lifespan startup
- Logs registry domain/provider counts

### 3) Health Visibility & API

Updated `src/api/health_models.py`:

- Added `HealthConfigCoreProviders`
- Added `config.core_providers` field in `HealthConfig`

Updated `src/api/health_utils.py`:

- Adds `config.core_providers` snapshot via `get_core_provider_registry_snapshot()`

Updated `src/api/v1/health.py`:

- Added response model `ProviderRegistryHealthResponse`
- Added endpoints:
  - `GET /api/v1/providers/registry`
  - `GET /api/v1/health/providers/registry`

## Tests Added/Updated

- Added `tests/unit/test_provider_framework_ocr_bridge.py`
  - OCR adapter process/health
  - trace override
  - invalid request rejection
  - OCR bootstrap registration
- Added `tests/unit/test_provider_registry_bootstrap.py`
  - bootstrap expected domains/providers
  - idempotence
  - snapshot shape
- Updated `tests/unit/test_health_hybrid_config.py`
  - verifies `config.core_providers` exists in health payload
  - verifies provider registry health endpoint snapshot

## Validation

Executed:

```bash
pytest tests/unit/test_provider_framework.py \
  tests/unit/test_provider_framework_vision_bridge.py \
  tests/unit/test_provider_framework_ocr_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_health_hybrid_config.py \
  tests/unit/test_main_coverage.py -q

python3 -m black src/core/providers src/api/health_models.py src/api/health_utils.py \
  src/api/v1/health.py src/main.py \
  tests/unit/test_provider_framework.py \
  tests/unit/test_provider_framework_vision_bridge.py \
  tests/unit/test_provider_framework_ocr_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_health_hybrid_config.py --check

python3 -m flake8 src/core/providers src/api/health_models.py src/api/health_utils.py \
  src/api/v1/health.py src/main.py \
  tests/unit/test_provider_framework.py \
  tests/unit/test_provider_framework_vision_bridge.py \
  tests/unit/test_provider_framework_ocr_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_health_hybrid_config.py --max-line-length=100
```

Results:

- Pytest: `44 passed, 1 warning`
- Black check: PASS
- Flake8: PASS

## Compatibility Notes

- Existing Vision/OCR runtime behavior remains unchanged; this is additive.
- Registry snapshot supports lazy bootstrap to remain robust in direct-function test paths.
