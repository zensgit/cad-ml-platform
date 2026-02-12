# DEV_PROVIDER_FRAMEWORK_CLASSIFIER_V16_V6_20260207

## Summary

Extended the classifier provider bridge (`src/core/providers/classifier.py`) to include **V16** and **V6** part classifiers as providers:

- `classifier/v16`
- `classifier/v6`

This makes these classifiers visible via:

- provider registry snapshot (`/api/v1/providers/registry`)
- provider readiness checks (`/api/v1/providers/health`)

## Why

- The system already contains V16/V6 DXF part classifiers (`src/ml/part_classifier.py`) and runtime fallbacks in `src/core/analyzer.py`.
- For ops/debug, we want a unified discovery layer (ProviderRegistry) so we can inspect availability and readiness without reading code paths.
- `torch` is optional in some environments; the provider adapters must avoid importing/loading heavy models during health checks.

## Changes

### 1) Provider Adapters

Updated `src/core/providers/classifier.py`:

- `ClassifierRequest` now supports `file_path` (optional).
- Added adapters:
  - `V16PartClassifierProviderAdapter`
  - `V6PartClassifierProviderAdapter`

Key behavior:

- **Health check is lightweight**:
  - uses `importlib.util.find_spec("torch")`
  - checks required model files exist
  - respects `DISABLE_V16_CLASSIFIER`
  - does not load models
- **Process** (`.process(...)`) expects `ClassifierRequest(file_path=...)` and returns a dict with:
  - `status`
  - `label`, `confidence`, `probabilities`
  - review fields for V16 when present

### 2) Registry Bootstrap

Updated `bootstrap_core_classifier_providers()` to register:

- `classifier/v16`
- `classifier/v6`

### 3) Tests

- Updated `tests/unit/test_provider_registry_bootstrap.py` to assert `v16` and `v6` are present.
- Updated `tests/unit/test_provider_framework_classifier_bridge.py` to assert providers are registered.

## Validation

Executed:

```bash
python3 -m pytest -q \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py

python3 -m flake8 \
  src/core/providers/classifier.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_provider_registry_bootstrap.py \
  --max-line-length=100

python3 -m mypy src/core/providers/classifier.py
```

Results:

- Pytest: PASS
- Flake8: PASS
- Mypy (targeted): PASS

## Notes

- These providers are **capability probes** for readiness; detailed V16 operational stats remain on the dedicated V16 endpoints under `src/api/v1/health.py`.

