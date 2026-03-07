# Qdrant Coarse Search Filters Validation

## Scope
- Extend vector search and similarity top-k APIs with coarse-contract filters
- Reuse stored label metadata instead of introducing a new search path
- Keep the change additive to the current request/response contracts

## Changed Files
- `src/api/v1/vectors.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_vectors_module_endpoints.py`
- `tests/unit/test_similarity_topk.py`

## Added Filters
- `fine_part_type_filter`
- `coarse_part_type_filter`
- `decision_source_filter`
- `is_coarse_label_filter`

## Validation
```bash
python3 -m py_compile src/api/v1/vectors.py src/api/v1/analyze.py \
  tests/unit/test_vectors_module_endpoints.py tests/unit/test_similarity_topk.py

flake8 src/api/v1/vectors.py src/api/v1/analyze.py \
  tests/unit/test_vectors_module_endpoints.py tests/unit/test_similarity_topk.py \
  --max-line-length=100

pytest -q tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py \
  tests/contract/test_openapi_schema_snapshot.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `13 passed`

## Verified Behavior
- `/api/v1/vectors/search` supports coarse/fine semantic filters
- `/api/v1/analyze/similarity/topk` supports coarse/fine semantic filters
- Decision source and coarse-label flag can both be used as filters
- Existing OpenAPI snapshot counts remain stable

## Notes
- This PR uses the existing metadata-backed search flow
- It does not yet switch the API path to a dedicated Qdrant-native filtered search implementation
