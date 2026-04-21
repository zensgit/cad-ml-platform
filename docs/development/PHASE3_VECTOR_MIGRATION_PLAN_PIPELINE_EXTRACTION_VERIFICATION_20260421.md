# Phase 3 Vector Migration Plan Pipeline Extraction Verification

日期：2026-04-21

## 本地验证
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_migration_plan_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_plan_pipeline.py tests/unit/test_vectors_migration_plan_delegation.py`
- `.venv311/bin/flake8 src/core/vector_migration_plan_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_migration_plan_pipeline.py tests/unit/test_vectors_migration_plan_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_migration_plan_pipeline.py tests/unit/test_vectors_migration_plan_delegation.py tests/unit/test_vector_migration_plan.py tests/unit/test_vector_migration_pending_summary.py tests/unit/test_vector_migration_pending.py`

## 预期验证点
- pending summary 的排序、比例、largest version 推导不漂移
- migration plan 的 batch 构造、partial scan 阻塞和推荐 payload 不漂移
- route 仍保持薄封装委托
