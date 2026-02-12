# DEV_PROVIDER_FRAMEWORK_CLASSIFIER_BRIDGE_20260207

## Summary

Extended the core provider framework (`src/core/providers/`) with a **classifier domain** bridge so DXF classifiers can be discovered and instantiated via `ProviderRegistry`.

This is intentionally lightweight and uses **lazy imports** to avoid failing on environments where optional ML dependencies (e.g. `torch`) are not installed.

## Changes

### 1) Classifier Provider Bridge

Added `src/core/providers/classifier.py`:

- `ClassifierRequest`: common request container (`filename`, optional `file_bytes`)
- `HybridClassifierProviderAdapter`: wraps `src/ml/hybrid_classifier.HybridClassifier`
- `Graph2DClassifierProviderAdapter`: wraps `src/ml/vision_2d.Graph2DClassifier` (or ensemble)
- `bootstrap_core_classifier_providers()` registers:
  - `classifier/hybrid`
  - `classifier/graph2d`
  - `classifier/graph2d_ensemble`

### 2) Bootstrap Wiring

Updated `src/core/providers/bootstrap.py` so `bootstrap_core_provider_registry()` includes the new classifier providers.

### 3) Public Exports

Updated `src/core/providers/__init__.py` to export:

- `ClassifierProviderConfig`, `ClassifierRequest`
- `HybridClassifierProviderAdapter`, `Graph2DClassifierProviderAdapter`
- `bootstrap_core_classifier_providers`

### 4) Tests

- Added `tests/unit/test_provider_framework_classifier_bridge.py`
- Updated `tests/unit/test_provider_registry_bootstrap.py` to assert classifier providers are present in the snapshot.

### 5) Type-Check Hygiene (Small Fixes)

- `src/__init__.py`: added return type for `__getattr__` for mypy.
- `src/utils/dxf_io.py`: removed unused `# type: ignore` on `import ezdxf`.

## Validation

Executed:

```bash
python3 -m pytest -q \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_provider_framework_classifier_bridge.py

python3 -m pytest -q \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_provider_framework.py \
  tests/unit/test_provider_framework_vision_bridge.py \
  tests/unit/test_provider_framework_ocr_bridge.py

python3 -m black \
  src/core/providers/classifier.py \
  src/core/providers/__init__.py \
  src/core/providers/bootstrap.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  --check

python3 -m flake8 \
  src/core/providers/classifier.py \
  src/core/providers/__init__.py \
  src/core/providers/bootstrap.py \
  tests/unit/test_provider_registry_bootstrap.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  --max-line-length=100

python3 -m mypy src/core/providers/classifier.py
```

Results:

- Pytest: PASS
- Black: PASS
- Flake8: PASS
- Mypy (targeted): PASS

## Notes

- `classifier/graph2d*` health checks intentionally do **not** parse DXF for readiness; they only reflect whether the underlying model is loaded (when available).
- This change is additive; existing `/api/v1/analyze` classification logic continues to work as-is.

