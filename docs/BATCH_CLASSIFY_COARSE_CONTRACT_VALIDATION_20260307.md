# Batch Classify Coarse Contract Validation - 2026-03-07

## Goal
Stabilize `/api/v1/analyze/batch-classify` so batch classifier consumers receive both exact and coarse labels without breaking the legacy `category` field.

## Scope
Updated files:
- `src/api/v1/analyze.py`
- `tests/unit/test_v16_classifier_endpoints.py`

## Contract Additions
Each `BatchClassifyResultItem` now exposes additive fields:
- `fine_category`
- `coarse_category`
- `is_coarse_label`
- `top2_category`
- `top2_confidence`

Legacy fields remain unchanged:
- `category`
- `confidence`
- `probabilities`
- `needs_review`
- `review_reason`
- `classifier`
- `error`

## Validation Commands
```bash
python3 -m py_compile src/api/v1/analyze.py tests/unit/test_v16_classifier_endpoints.py
flake8 src/api/v1/analyze.py tests/unit/test_v16_classifier_endpoints.py --max-line-length=100
pytest -q tests/unit/test_v16_classifier_endpoints.py
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `18 passed`

## Notes
- V16 batch predictions now surface top-2 values directly in the batch response.
- V6 sequential fallback now exposes the same fine/coarse contract.
- The old `category` field remains the exact prediction for backward compatibility.
