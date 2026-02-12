# DEV_PROVIDER_COVERAGE_TESTS_REGISTRY_VISION_OCR_20260209

## Goal
Expand provider-framework unit coverage for:
- Provider registry semantics (`ProviderRegistry`)
- Vision provider adapter + bootstrap wiring
- OCR provider adapter + bootstrap wiring

## Changes
### ProviderRegistry coverage suite
File: `tests/unit/test_registry_coverage.py`
- Covers `unregister()` edge cases (nonexistent provider, cached-instance cleanup).
- Covers cache enable/disable behavior via `PROVIDER_REGISTRY_CACHE_ENABLED`.
- Covers `clear_instances()` vs `clear()` semantics.
- Covers duplicate registration errors and listing helpers (`list_domains`, `list_providers`).

### Vision provider adapter coverage suite
File: `tests/unit/test_vision_provider_coverage.py`
- Covers strict input validation for `VisionProviderAdapter.process()`:
  - rejects non-bytes, rejects empty bytes
  - accepts `bytearray`
- Covers `health_check()` bridging behavior:
  - sync `health_check()`
  - async `health_check()`
  - fallback to `analyze_image()` probe when no health method exists
- Covers `bootstrap_core_vision_providers()` registrations and default config wiring.

### OCR provider adapter coverage suite (partial)
File: `tests/unit/test_ocr_provider_coverage.py`
- Covers strict input validation for `OcrProviderAdapter.process()`:
  - rejects non-bytes, rejects empty bytes
  - accepts `bytearray`
- Covers `health_check()` bridging behavior (sync/async/no-method).
- Covers `warmup()` bridging behavior (sync/async/no-method).
- Covers `extract()` compatibility method (`extract(...) -> process(...)`).
- Covers `_build_default_provider` unsupported-provider error path.
- Covers `bootstrap_core_ocr_providers()` registrations and idempotency.

Note:
- OCR adapter branches that instantiate real providers (`paddle`, `deepseek_hf`)
  remain intentionally uncovered in unit tests to avoid importing heavyweight
  runtime dependencies in a minimal test environment.

## Validation
Commands run:
```bash
pytest tests/unit/test_registry_coverage.py -q
pytest tests/unit/test_vision_provider_coverage.py -q
pytest tests/unit/test_ocr_provider_coverage.py -q

flake8 tests/unit/test_registry_coverage.py \
  tests/unit/test_vision_provider_coverage.py \
  tests/unit/test_ocr_provider_coverage.py
```

Coverage spot checks:
```bash
pytest tests/unit/test_registry_coverage.py --cov=src.core.providers.registry --cov-report=term-missing -q
pytest tests/unit/test_vision_provider_coverage.py --cov=src.core.providers.vision --cov-report=term-missing -q
pytest tests/unit/test_ocr_provider_coverage.py --cov=src.core.providers.ocr --cov-report=term-missing -q
```

Results:
- `src/core/providers/registry.py`: 100% in spot check
- `src/core/providers/vision.py`: 100% in spot check
- `src/core/providers/ocr.py`: 87% in spot check (missing branches that instantiate real OCR providers)

