# Vector Search Coarse Contract Validation

## Scope

Extend vector-backed retrieval surfaces so that list, search, and top-k
similarity results expose the same stable fine/coarse semantic contract as
batch similarity and compare.

Covered endpoints:

- `GET /api/v1/vectors`
- `POST /api/v1/vectors/search`
- `POST /api/v1/analyze/similarity/topk`

## Changes

### 1. Vector list contract

`src/api/v1/vectors.py` `VectorListItem` now includes additive fields:

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `decision_source`
- `is_coarse_label`

These fields are populated for both memory and redis-backed vector listing.

### 2. Vector search contract

`POST /api/v1/vectors/search` now returns the same additive fields per result.

### 3. Similarity top-k contract

`src/api/v1/analyze.py` `SimilarityTopKItem` now includes the same additive
fields per result.

All three surfaces reuse `extract_vector_label_contract(meta)` from
`src/core/similarity.py`.

## Tests

```bash
python3 -m py_compile \
  src/api/v1/vectors.py \
  src/api/v1/analyze.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py

flake8 \
  src/api/v1/vectors.py \
  src/api/v1/analyze.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `10 passed`

Validated behaviors:

1. Memory-backed vector listing returns coarse/fine semantic metadata.
2. Redis-backed vector listing returns coarse/fine semantic metadata.
3. Vector search returns coarse/fine semantic metadata.
4. Similarity top-k returns coarse/fine semantic metadata.

## Compatibility

- Existing response fields remain unchanged.
- New semantic fields are additive.
- No change to ranking, filtering, or vector selection logic.
