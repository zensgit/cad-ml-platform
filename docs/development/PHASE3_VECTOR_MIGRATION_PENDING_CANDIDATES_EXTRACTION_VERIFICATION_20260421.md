# Phase 3 Vector Migration Pending Candidates Extraction Verification

## Local Validation

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_migration_pending_candidates.py src/api/v1/vectors.py tests/unit/test_vector_migration_pending_candidates.py tests/unit/test_vectors_migration_pending_candidates_delegation.py`
- `.venv311/bin/flake8 src/core/vector_migration_pending_candidates.py src/api/v1/vectors.py tests/unit/test_vector_migration_pending_candidates.py tests/unit/test_vectors_migration_pending_candidates_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migration_pending_candidates.py tests/unit/test_vectors_migration_pending_candidates_delegation.py tests/unit/test_vector_migration_pending.py tests/unit/test_vector_migration_pending_summary.py tests/unit/test_vector_migration_plan.py tests/unit/test_vector_migration_pending_run.py tests/unit/test_vectors_migration_plan_delegation.py tests/unit/test_vectors_migration_pending_run_delegation.py`

## Expected Outcome

- Shared helper owns the pending-candidate collection logic
- `vectors.py` keeps a thin compatibility wrapper for existing monkeypatch surfaces
- Existing pending/summary/plan/run route behavior remains unchanged
