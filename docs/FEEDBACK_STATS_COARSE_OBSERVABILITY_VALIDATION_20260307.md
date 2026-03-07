# Feedback Stats Coarse Observability Validation

## Scope
- Add `GET /api/v1/feedback/stats`
- Aggregate coarse/fine correction outcomes from the feedback log
- Preserve the existing feedback submission contract
- Refresh the OpenAPI contract snapshot for the new endpoint

## Changed Files
- `src/api/v1/feedback.py`
- `tests/test_feedback.py`
- `config/openapi_schema_snapshot.json`

## Validation
```bash
python3 -m py_compile src/api/v1/feedback.py tests/test_feedback.py

flake8 src/api/v1/feedback.py tests/test_feedback.py --max-line-length=100

python3 scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json

pytest -q tests/test_feedback.py tests/contract/test_openapi_schema_snapshot.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- OpenAPI snapshot regenerated intentionally
- `pytest`: `6 passed`

## Verified Behavior
- Missing feedback log returns an empty stats payload instead of an error
- Stats aggregate:
  - `review_outcome`
  - `review_reasons`
  - `corrected_fine_part_type`
  - `corrected_coarse_part_type`
  - `original_fine_part_type`
  - `original_coarse_part_type`
  - `original_decision_source`
- Stats compute:
  - `correction_count`
  - `coarse_correction_count`
  - `average_rating`

## Notes
- This is additive: the existing `POST /api/v1/feedback/` behavior is unchanged
- OpenAPI snapshot count changes are expected because a new route was added
