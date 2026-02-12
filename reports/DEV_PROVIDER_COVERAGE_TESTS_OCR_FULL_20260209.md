# DEV_PROVIDER_COVERAGE_TESTS_OCR_FULL_20260209

## Goal
Complete unit-test coverage for the provider-framework OCR bridge:
- `src/core/providers/ocr.py`

Specifically, cover the remaining default-provider import/instantiation branches
and the bootstrap default config wiring, without importing heavyweight OCR
runtime dependencies.

## Changes
File: `tests/unit/test_ocr_provider_coverage.py`
- Added a helper `_fake_ocr_provider_module(...)` that injects lightweight module
  stubs into `sys.modules` for:
  - `src.core.ocr.providers.paddle` (`PaddleOcrProvider`)
  - `src.core.ocr.providers.deepseek_hf` (`DeepSeekHfProvider`)
- Added tests to cover:
  - `_build_default_provider` branches for `paddle` and `deepseek_hf`
  - `bootstrap_core_ocr_providers()` default config construction at instantiation
    time (via `ProviderRegistry.get(...)`)

Rationale:
- The real provider implementations may import large dependencies (or require
  system libraries). Unit tests should keep these paths isolated while still
  covering the adapter/registry logic.

## Validation
Commands run:
```bash
pytest tests/unit/test_ocr_provider_coverage.py -q
pytest tests/unit/test_ocr_provider_coverage.py --cov=src.core.providers.ocr --cov-report=term-missing -q
flake8 tests/unit/test_ocr_provider_coverage.py
```

Results:
- Unit tests: passed
- Coverage spot check:
  - `src/core/providers/ocr.py`: 100% (0 missing)

