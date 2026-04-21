# Phase 3 Vector Migration Preview Pipeline Extraction Verification

日期：2026-04-21

## 本地验证
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_migration_preview_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_preview_pipeline.py tests/unit/test_vectors_migration_preview_delegation.py`
- `.venv311/bin/flake8 src/core/vector_migration_preview_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_preview_pipeline.py tests/unit/test_vectors_migration_preview_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migration_preview_pipeline.py tests/unit/test_vectors_migration_preview_delegation.py tests/unit/test_migration_preview_trends.py tests/unit/test_migration_preview_stats.py tests/unit/test_migration_preview_response.py tests/unit/test_vector_migrate_layouts.py`

## 预期验证点
- invalid version 仍返回 `422 INPUT_VALIDATION_FAILED`
- qdrant preview 的版本分布、limit、with_vectors 行为不漂移
- memory preview 的 limit cap 仍为 `100`
- route 仍保持薄封装委托
