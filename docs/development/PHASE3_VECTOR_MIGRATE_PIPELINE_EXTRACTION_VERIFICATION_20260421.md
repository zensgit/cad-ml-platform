# Phase 3 Vector Migrate Pipeline Extraction Verification

## Local Validation

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_migrate_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migrate_pipeline.py tests/unit/test_vectors_migrate_delegation.py`
- `.venv311/bin/flake8 src/core/vector_migrate_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migrate_pipeline.py tests/unit/test_vectors_migrate_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migrate_pipeline.py tests/unit/test_vectors_migrate_delegation.py tests/unit/test_vector_migrate_api.py tests/unit/test_vector_migrate_metrics.py tests/unit/test_vector_migrate_dimension_mismatch.py tests/unit/test_vector_migrate_response_fields.py tests/unit/test_vector_migration_history.py tests/unit/test_vector_migration_counts_history.py tests/unit/test_vector_migrate_downgrade_chain.py tests/unit/test_vector_migrate_v4.py`

## Expected Outcome

- Shared helper owns the main vector migration body
- `vectors.py` remains a thin route wrapper with stable patch surfaces
- API behavior, history ring buffer, metrics, dry-run, and downgrade semantics remain unchanged
