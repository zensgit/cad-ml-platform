# Analyze Similarity Coarse Contract Validation

## Scope

Extend `POST /api/v1/analyze/similarity` so the response includes stable
coarse/fine semantic metadata for both the reference vector and the target
vector.

## Changes

`src/api/v1/analyze.py` `SimilarityResult` now exposes additive fields:

- `reference_part_type`
- `reference_fine_part_type`
- `reference_coarse_part_type`
- `reference_decision_source`
- `reference_is_coarse_label`
- `target_part_type`
- `target_fine_part_type`
- `target_coarse_part_type`
- `target_decision_source`
- `target_is_coarse_label`

These fields are resolved from vector metadata using
`extract_vector_label_contract(meta)`.

## Tests

```bash
python3 -m py_compile \
  src/api/v1/analyze.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py

flake8 \
  src/api/v1/analyze.py \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_similarity_endpoint.py \
  tests/unit/test_similarity_error_codes.py
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed

Validated behaviors:

1. The similarity endpoint still returns normal score and error statuses.
2. The similarity endpoint now exposes reference and target coarse/fine labels
   when vector metadata is available.
3. Existing `reference_not_found` and `dimension_mismatch` flows remain intact.
