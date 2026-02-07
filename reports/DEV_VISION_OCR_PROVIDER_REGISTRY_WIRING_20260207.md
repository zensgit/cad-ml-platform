# DEV_VISION_OCR_PROVIDER_REGISTRY_WIRING_20260207

## Summary

Aligned the Vision API OCR integration with the new core provider framework:

- `src/api/v1/vision.py` now builds its `OcrManager` providers via `bootstrap_core_provider_registry()` + `ProviderRegistry.get(...)` (best-effort).
- `OcrManager` now normalizes common strategy aliases (`deepseek` -> `deepseek_hf`) and case-folds explicit provider names.

This makes Vision's optional OCR path consistent with `/api/v1/ocr` and the Analyze optional OCR block.

## Changes

### 1) Normalize OCR strategy names

Updated `src/core/ocr/manager.py`:

- Normalizes strategy input using `strip().lower()`.
- Maps aliases:
  - `deepseek` -> `deepseek_hf`
  - `deepseek-hf` -> `deepseek_hf`

This prevents accidental `PROVIDER_DOWN` errors when clients send older/shorter names (e.g. `deepseek`).

### 2) Wire Vision OCRManager providers from ProviderRegistry

Updated `src/api/v1/vision.py`:

- Calls `bootstrap_core_provider_registry()` when building a new singleton manager.
- Registers OCR providers on the created `OcrManager`:
  - `paddle`
  - `deepseek_hf`
- Uses best-effort registration (exceptions are swallowed) to preserve existing graceful-degradation behavior.

### 3) Align Vision request docs with actual provider names

Updated `src/core/vision/base.py`:

- Adjusted the `ocr_provider` field description to reference `deepseek_hf` (and mention `deepseek` as an alias).

### 4) Fix provider framework module docstring drift

Updated `src/core/providers/__init__.py`:

- Removed mention of a non-existent `ConfigurableProvider`.

### 5) Tests

Added:

- `tests/unit/test_ocr_manager_strategy_aliases.py`
- `tests/unit/test_vision_api_provider_registry_wiring.py`

## Validation

Executed:

```bash
pytest -q \
  tests/unit/test_ocr_manager_strategy_aliases.py \
  tests/unit/test_vision_api_provider_registry_wiring.py \
  tests/unit/test_vision_api_coverage.py \
  tests/unit/test_ocr_endpoint_coverage.py

black --check \
  src/api/v1/vision.py \
  src/core/ocr/manager.py \
  src/core/vision/base.py \
  src/core/providers/__init__.py \
  tests/unit/test_ocr_manager_strategy_aliases.py \
  tests/unit/test_vision_api_provider_registry_wiring.py

flake8 \
  src/api/v1/vision.py \
  src/core/ocr/manager.py \
  src/core/vision/base.py \
  src/core/providers/__init__.py \
  tests/unit/test_ocr_manager_strategy_aliases.py \
  tests/unit/test_vision_api_provider_registry_wiring.py \
  --max-line-length=100
```

Results:

- Pytest: `45 passed, 1 warning`
- Black: PASS
- Flake8: PASS

## Notes

- `scripts/test_with_local_api.sh` is a supported tiered local runner (referenced by `Makefile` and docs); it is not a throwaway script.
