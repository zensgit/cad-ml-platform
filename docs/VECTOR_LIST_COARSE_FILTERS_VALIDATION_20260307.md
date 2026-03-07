# Vector List Coarse Filters Validation

## Scope
- Extend `GET /api/v1/vectors` with coarse/fine semantic filters
- Keep pagination behavior stable
- Return filtered `total` counts without changing unfiltered behavior

## Changed Files
- `src/api/v1/vectors.py`
- `tests/unit/test_vectors_module_endpoints.py`

## Added Query Params
- `material_filter`
- `complexity_filter`
- `fine_part_type_filter`
- `coarse_part_type_filter`
- `decision_source_filter`
- `is_coarse_label_filter`

## Validation
```bash
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vectors_module_endpoints.py

flake8 src/api/v1/vectors.py \
  tests/unit/test_vectors_module_endpoints.py \
  --max-line-length=100

pytest -q tests/unit/test_vectors_module_endpoints.py \
  tests/contract/test_openapi_schema_snapshot.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `12 passed`

## Verified Behavior
- Memory-backed vector listing supports coarse-contract filters
- Redis-backed vector listing supports coarse-contract filters
- `total` counts all matched rows, while `vectors` respects `offset/limit`
- Existing OpenAPI snapshot counts remain stable

## Notes
- This PR is stacked on the coarse search filters branch
- Filtering is metadata-backed and additive to the current list endpoint
