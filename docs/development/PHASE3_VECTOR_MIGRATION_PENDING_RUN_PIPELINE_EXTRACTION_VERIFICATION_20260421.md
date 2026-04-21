# Phase 3 Vector Migration Pending Run Pipeline Extraction Verification

日期：2026-04-21

## 本地验证
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_migration_pending_run_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_pending_run_pipeline.py tests/unit/test_vectors_migration_pending_run_delegation.py`
- `.venv311/bin/flake8 src/core/vector_migration_pending_run_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_pending_run_pipeline.py tests/unit/test_vectors_migration_pending_run_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migration_pending_run_pipeline.py tests/unit/test_vectors_migration_pending_run_delegation.py tests/unit/test_vector_migration_pending_run.py`

## 预期验证点
- qdrant partial scan 且未显式 override 时仍返回 `409 CONSTRAINT_VIOLATION`
- `allow_partial_scan=true` 时仍会继续调用 `migrate_vectors()`
- `from_version_filter` 透传语义不漂移
- route 仍保持薄封装委托
