# DEV_PROVIDER_REGISTRY_RUNTIME_NORMALIZATION_20260213

## Summary

Hardened the core provider framework by unifying provider ID normalization and validation
across runtime registry operations, and aligned readiness provider-id parsing to the same
rules used by registration.

## Problem

- `ProviderRegistry.register()` validated and normalized `(domain, provider_name)`.
- Runtime methods (`get_provider_class`, `get`, `exists`, `list_providers`, `unregister`)
  previously did not use the same normalization path.
- Resulting risks:
  - Inconsistent behavior for whitespace-wrapped IDs.
  - Different acceptance/rejection semantics between registration and runtime lookups.
  - Potentially malformed provider IDs entering readiness config parsing.

## Changes

- `src/core/providers/registry.py`
  - Added `ProviderRegistry.normalize_provider_id(domain, provider_name)`.
  - Updated runtime methods to use normalized IDs consistently:
    - `get_provider_class`
    - `get`
    - `exists`
    - `list_providers` (domain normalization)
    - `unregister`
  - Runtime methods now reject invalid separators (`/`, `:`) consistently with registration.

- `src/core/providers/readiness.py`
  - Updated `parse_provider_id_list()` to reuse
    `ProviderRegistry.normalize_provider_id(...)`.
  - Invalid parsed IDs are ignored deterministically.

- Tests
  - `tests/unit/test_registry_coverage.py`
    - Added `TestRuntimeTokenNormalization` covering:
      - whitespace-normalized runtime lookup/cache/unregister behavior
      - runtime rejection of invalid separator tokens
  - `tests/unit/test_readiness_coverage.py`
    - Added parser regression for tokens with extra separators
      (`vision/openai/v2`, `ocr:deep:extra`) to ensure invalid IDs are ignored.

## Validation

Executed:

```bash
.venv/bin/python -m pytest tests/unit/test_registry_coverage.py tests/unit/test_readiness_coverage.py tests/unit/test_provider_framework.py -v
make validate-core-fast
```

Results:

- Targeted provider framework tests: `69 passed`
- `make validate-core-fast`: passed
  - tolerance checks/tests
  - OpenAPI operation/schema contracts
  - service-mesh tests
  - provider core tests
  - provider contract tests

## Risk Assessment

- Low risk:
  - Change is internal to provider ID handling and keeps existing valid IDs unchanged.
  - Added regression coverage for new normalization and parser behavior.
  - Full fast validation suite passed.

