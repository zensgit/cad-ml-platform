# Vectors Stats Coarse Observability Validation

## Scope

Extend `vectors_stats` endpoints so they expose coarse semantic and decision
source distributions in addition to material/complexity/format counts.

Covered endpoints:

- `GET /api/v1/vectors_stats/stats`
- `GET /api/v1/vectors_stats/distribution`

## Changes

### `VectorStatsResponse`

Added:

- `by_coarse_part_type`
- `by_decision_source`

### `VectorDistributionResponse`

Added:

- `by_coarse_part_type`
- `by_decision_source`
- `dominant_coarse_ratio`

### Summary logic

`src/api/v1/vectors_stats.py` now derives these fields from vector metadata via
`extract_vector_label_contract(meta)` for both memory and redis backends.

## Tests

```bash
python3 -m py_compile \
  src/api/v1/vectors_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_vector_stats.py

flake8 \
  src/api/v1/vectors_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_vector_stats.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_vector_stats.py
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed

Validated behaviors:

1. Memory-backed stats include coarse part type and decision source counts.
2. Redis-backed stats include coarse part type and decision source counts.
3. Distribution endpoint exposes `dominant_coarse_ratio`.
