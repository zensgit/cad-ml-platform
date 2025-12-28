# DEV_BATCH_SIMILARITY_FAISS_UNAVAILABLE_DEGRADED_FLAG_FIX_20251229

## Scope
- Verify batch similarity degraded flag remains boolean when Faiss backend is unavailable.

## Changes
- `src/api/v1/vectors.py`
  - Detect fallback using store metadata to avoid env drift.
- `src/core/similarity.py`
  - Attach requested/actual backend metadata when constructing stores.

## Validation
- Command: `source .venv/bin/activate && pytest tests/unit/test_batch_similarity_faiss_unavailable.py::test_batch_similarity_faiss_unavailable_degraded_flag -q`
  - Result: 1 passed in 1.92s (Python 3.11.13).
