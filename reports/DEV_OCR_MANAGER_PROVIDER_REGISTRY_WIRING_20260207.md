# DEV_OCR_MANAGER_PROVIDER_REGISTRY_WIRING_20260207

## Summary

Switched OCR provider construction in API paths to use the new core `ProviderRegistry`, so the provider framework is used in real runtime code (not only for health visibility).

## Changes

### 1) Make OCR adapter usable as an `OcrClient`

Updated `src/core/providers/ocr.py`:

- Added `warmup()` passthrough to wrapped provider when available.
- Added `extract(image_bytes, trace_id)` method that delegates to `process(...)`.

This enables using `OcrProviderAdapter` instances wherever an `OcrClient` is expected.

### 2) Wire `/api/v1/ocr` manager to `ProviderRegistry`

Updated `src/api/v1/ocr.py`:

- Replaced direct instantiation (`PaddleOcrProvider()` / `DeepSeekHfProvider()`) with:
  - `bootstrap_core_provider_registry()`
  - `ProviderRegistry.get('ocr', 'paddle')`
  - `ProviderRegistry.get('ocr', 'deepseek_hf')`

### 3) Wire Analyze optional OCR path

Updated `src/api/v1/analyze.py` optional OCR integration block:

- Uses the same registry bootstrap + provider acquisition pattern.

### 4) Tests

Updated `tests/unit/test_ocr_endpoint_coverage.py`:

- Strengthened `get_manager` coverage to assert the expected provider names are registered.

## Validation

Executed:

```bash
pytest tests/unit/test_provider_framework_ocr_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_ocr_endpoint_coverage.py -q

python3 -m black src/core/providers src/api/v1/ocr.py src/api/v1/analyze.py \
  tests/unit/test_ocr_endpoint_coverage.py --check

python3 -m flake8 src/core/providers src/api/v1/ocr.py src/api/v1/analyze.py \
  tests/unit/test_ocr_endpoint_coverage.py --max-line-length=100
```

Results:

- Pytest: `29 passed, 1 warning`
- Black: PASS
- Flake8: PASS

## Notes

- This is intended to be behavior-preserving: same two OCR backends remain registered (`paddle`, `deepseek_hf`).
- Provider selection and fallback remain in `OcrManager`.
