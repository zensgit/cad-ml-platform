# DEV_BATCH_SIMILARITY_FAISS_UNAVAILABLE_20251224

## Scope
- Validate batch similarity fallback path when Faiss backend is unavailable.

## Changes
- `tests/unit/test_batch_similarity_faiss_unavailable.py`
  - Implemented degraded/fallback assertions with forced Faiss unavailability.

## Validation
- Command: `.venv/bin/python -m pytest tests/unit/test_batch_similarity_faiss_unavailable.py -v`
  - Result: 1 passed.
