# DEV_BATCH_SIMILARITY_FAISS_UNAVAILABLE_DEGRADED_FLAG_FIX_20251229

## Scope
- Verify batch similarity degraded flag remains boolean when Faiss backend is unavailable.

## Changes
- `src/api/v1/vectors.py`
  - Normalize degraded/fallback computation to guard against null degraded info.

## Validation
- Command: `source .venv/bin/activate && pytest tests/unit/test_batch_similarity_faiss_unavailable.py::test_batch_similarity_faiss_unavailable_degraded_flag -q`
  - Result: 1 passed in 6.66s (Python 3.11.13).
