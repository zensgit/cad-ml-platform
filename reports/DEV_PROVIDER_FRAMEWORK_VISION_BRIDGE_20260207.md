# DEV_PROVIDER_FRAMEWORK_VISION_BRIDGE_20260207

## Summary

Integrated the new core provider framework with the existing vision subsystem through an additive bridge layer, without changing current vision runtime flows.

## Changes

### 1) Vision bridge module

Added `src/core/providers/vision.py`:

- `VisionProviderConfig` (extends `ProviderConfig`)
  - `provider_name`
  - `include_description_default`
  - `provider_kwargs`
- `VisionProviderAdapter` (extends `BaseProvider`)
  - wraps existing `src.core.vision.base.VisionProvider`
  - `process(bytes)` -> delegates to `analyze_image(...)`
  - validates input type and non-empty bytes
  - health check strategy:
    - use wrapped provider `health_check` if available
    - otherwise run lightweight analyze probe
- `bootstrap_core_vision_providers()`
  - registers `vision/stub`
  - registers `vision/deepseek_stub`
  - both backed by `DeepSeekStubProvider`

### 2) Core exports

Updated `src/core/providers/__init__.py` to export:

- `VisionProviderConfig`
- `VisionProviderAdapter`
- `bootstrap_core_vision_providers`

### 3) Unit tests

Added `tests/unit/test_provider_framework_vision_bridge.py`:

- adapter process + health check
- invalid request type rejection
- bootstrap registration and registry-based instance creation

## Validation

Commands executed:

```bash
pytest tests/unit/test_provider_framework.py tests/unit/test_provider_framework_vision_bridge.py -q
python3 -m black src/core/providers tests/unit/test_provider_framework.py tests/unit/test_provider_framework_vision_bridge.py --check
python3 -m flake8 src/core/providers tests/unit/test_provider_framework.py tests/unit/test_provider_framework_vision_bridge.py --max-line-length=100
```

Results:

- tests: PASS (`10 passed`)
- formatting: PASS
- lint: PASS

## Compatibility

- Existing vision/ocr orchestration remains unchanged.
- This is additive and provides a migration path toward unified provider registration.
