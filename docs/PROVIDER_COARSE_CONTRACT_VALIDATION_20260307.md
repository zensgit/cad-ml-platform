# Provider Coarse Contract Validation - 2026-03-07

## Goal
Stabilize classifier provider payloads so downstream callers can rely on:
- `fine_label`
- `coarse_label`
- `is_coarse_label`
- `decision_source`
- review governance fields when available

## Scope
Updated provider adapters:
- `HybridClassifierProviderAdapter`
- `Graph2DClassifierProviderAdapter`
- `V16PartClassifierProviderAdapter`
- `V6PartClassifierProviderAdapter`

Updated tests:
- `tests/unit/test_provider_framework_classifier_bridge.py`
- `tests/unit/test_classifier_provider_coverage.py`

## Implementation Summary
- Added a shared payload augmentation helper in `src/core/providers/classifier.py`
- Normalized decision-source serialization
- Added coarse/fine label contract fields without breaking existing `label` payloads
- Passed through review-governance fields when the wrapped hybrid result exposes them

## Validation Commands
```bash
python3 -m py_compile \
  src/core/providers/classifier.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_classifier_provider_coverage.py

flake8 \
  src/core/providers/classifier.py \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_classifier_provider_coverage.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_provider_framework_classifier_bridge.py \
  tests/unit/test_classifier_provider_coverage.py
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `30 passed`

## Notes
- Backward compatibility is preserved: legacy `label` fields remain unchanged.
- The new fields are additive and safe for existing consumers.
- Hybrid review-governance fields are optional passthroughs, so this adapter remains compatible with older wrapped result objects.
