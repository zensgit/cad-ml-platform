# Qdrant Native Compare Validation 2026-03-07

## Scope
- Route `POST /api/compare`
- Route `POST /api/v1/analyze/similarity`
- `analyze` internal vector registration and inline similarity when `VECTOR_STORE_BACKEND=qdrant`

## Changes
- Added local Qdrant backend helper in:
  - `src/api/v1/compare.py`
  - `src/api/v1/analyze.py`
- `compare` now fetches reference vector and metadata from Qdrant when Qdrant backend is enabled.
- `analyze` now registers analysis vectors to Qdrant instead of only the legacy in-memory store when Qdrant backend is enabled.
- `analyze` inline similarity (`calculate_similarity/reference_id`) now resolves the reference vector from Qdrant when available.
- `POST /api/v1/analyze/similarity` now computes cosine similarity from Qdrant vectors when Qdrant backend is enabled, while preserving the existing response contract.

## Validation Commands
```bash
python3 -m py_compile \
  src/api/v1/compare.py \
  src/api/v1/analyze.py \
  tests/unit/test_compare_endpoint.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py

flake8 \
  src/api/v1/compare.py \
  src/api/v1/analyze.py \
  tests/unit/test_compare_endpoint.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_compare_endpoint.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py
```

## Expected Coverage
- Qdrant compare success path
- Qdrant analyze registration -> similarity path
- Qdrant reference-not-found path
- Qdrant dimension-mismatch path

## Notes
- The legacy in-memory path remains unchanged as fallback.
- This change closes the main contract gap left after vector list/search/topk Qdrant support was merged.
