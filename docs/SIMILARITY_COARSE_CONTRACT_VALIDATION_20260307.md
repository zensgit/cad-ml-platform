# Similarity Coarse Contract Validation

## Scope

Stabilize coarse/fine semantic fields across vector-backed similarity surfaces:

- `POST /api/v1/analyze/` vector registration metadata
- `POST /api/v1/vectors/similarity/batch`
- `POST /api/compare`

## Changes

### 1. Vector metadata registration

`src/api/v1/analyze.py` now writes semantic contract fields into `_VECTOR_META`
when an analysis registers a feature vector:

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `final_decision_source`
- `is_coarse_label`

### 2. Shared vector metadata contract helper

`src/core/similarity.py` now exposes `extract_vector_label_contract(meta)` to
derive stable response fields from vector metadata with coarse-label fallback.

### 3. Batch similarity response enrichment

`src/api/v1/vectors.py` now includes the following per similar item when
metadata is available:

- `part_type`
- `fine_part_type`
- `coarse_part_type`
- `decision_source`
- `is_coarse_label`

### 4. Compare response enrichment

`src/api/v1/compare.py` now includes additive reference metadata fields:

- `reference_part_type`
- `reference_fine_part_type`
- `reference_coarse_part_type`
- `reference_decision_source`
- `reference_is_coarse_label`

## Tests

### Static validation

```bash
python3 -m py_compile \
  src/core/similarity.py \
  src/api/v1/analyze.py \
  src/api/v1/vectors.py \
  src/api/v1/compare.py \
  tests/unit/test_batch_similarity_empty_results.py \
  tests/unit/test_compare_endpoint.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py

flake8 \
  src/core/similarity.py \
  src/api/v1/analyze.py \
  src/api/v1/vectors.py \
  src/api/v1/compare.py \
  tests/unit/test_batch_similarity_empty_results.py \
  tests/unit/test_compare_endpoint.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  --max-line-length=100
```

### Runtime validation

```bash
pytest -q \
  tests/unit/test_batch_similarity_empty_results.py \
  tests/unit/test_compare_endpoint.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `18 passed`

Validated behaviors:

1. Batch similarity returns coarse/fine semantic metadata for similar vectors.
2. Compare endpoint exposes reference coarse/fine semantic metadata.
3. Analyze endpoint persists coarse/fine semantic metadata into vector store
   registration, so later similarity and compare calls can reuse it.

## Compatibility

- Existing response fields are preserved.
- New fields are additive and optional.
- No change to vector similarity scoring or filtering behavior.
