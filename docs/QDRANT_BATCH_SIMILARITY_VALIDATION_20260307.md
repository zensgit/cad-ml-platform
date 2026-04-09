# Qdrant Batch Similarity Validation 2026-03-07

## Scope
- Add native Qdrant handling for `POST /api/v1/vectors/similarity/batch`
- Preserve existing in-memory and Faiss fallback behavior
- Keep response contract unchanged

## Changes
- `src/api/v1/vectors.py`
  - Detect Qdrant backend via `_get_qdrant_store_or_none()`
  - For each requested vector id:
    - load source vector from Qdrant with `get_vector()`
    - search neighbors with `search_similar()`
    - apply existing response contract fields:
      - `part_type`
      - `fine_part_type`
      - `coarse_part_type`
      - `decision_source`
      - `is_coarse_label`
  - Preserve memory/Faiss path as fallback branch
- `tests/unit/test_batch_similarity.py`
  - add Qdrant success-path test
  - add Qdrant not-found test

## Validation
### Static
```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_batch_similarity.py
flake8 src/api/v1/vectors.py tests/unit/test_batch_similarity.py --max-line-length=100
```

### Tests
```bash
pytest -q \
  tests/unit/test_batch_similarity.py::test_batch_similarity_uses_qdrant_when_enabled \
  tests/unit/test_batch_similarity.py::test_batch_similarity_qdrant_not_found

pytest -q \
  tests/unit/test_batch_similarity.py \
  tests/unit/test_batch_similarity_faiss_unavailable.py \
  tests/unit/test_batch_similarity_empty_results.py
```

## Results
- Targeted Qdrant tests: `2 passed`
- Batch similarity regression suite: `29 passed`
- No lint or compile failures

## Notes
- This change does not alter migration/report endpoints.
- Qdrant path is intentionally isolated to batch similarity only.
