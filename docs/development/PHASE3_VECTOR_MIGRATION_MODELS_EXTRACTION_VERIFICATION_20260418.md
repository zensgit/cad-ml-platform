# Phase 3 Vector Migration Models Extraction Verification

## Implemented
- added shared schema module: `src/api/v1/vector_migration_models.py`
- updated `src/api/v1/vectors.py` to import the shared migration request/response models
- updated `src/api/v1/analyze_aux_models.py` to reuse the same migration schemas
- added smoke coverage: `tests/unit/test_vector_migration_models.py`

## Validation
- `python3 -m py_compile src/api/v1/vector_migration_models.py src/api/v1/vectors.py src/api/v1/analyze_aux_models.py tests/unit/test_vector_migration_models.py`
- `.venv311/bin/flake8 src/api/v1/vector_migration_models.py src/api/v1/vectors.py src/api/v1/analyze_aux_models.py tests/unit/test_vector_migration_models.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migration_models.py tests/unit/test_analyze_aux_models.py tests/unit/test_vector_migrate_layouts.py tests/unit/test_migration_preview_trends.py tests/unit/test_vector_migration_status.py tests/unit/test_vector_migrate_response_fields.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py`

## Result
- static validation passed
- vectors migration and legacy analyze migration regressions passed
- analyze and vectors routes now share one migration schema contract
