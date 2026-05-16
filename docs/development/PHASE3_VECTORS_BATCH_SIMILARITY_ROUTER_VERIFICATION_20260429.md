# Phase 3 Vectors Batch Similarity Router Verification

日期：2026-04-29

## 变更摘要

本次将 `POST /api/v1/vectors/similarity/batch` 从 `src/api/v1/vectors.py` 拆到独立 router：

- 新增 `src/api/v1/vector_similarity_models.py`
- 新增 `src/api/v1/vectors_similarity_router.py`
- 新增 `tests/unit/test_vectors_similarity_router.py`
- 更新 `src/api/v1/vectors.py` include similarity router，并继续 re-export endpoint/model

## 本地验证命令

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_similarity_router_pycache python3 -m py_compile src/api/v1/vectors.py src/api/v1/vector_similarity_models.py src/api/v1/vectors_similarity_router.py`
- `.venv311/bin/flake8 src/api/v1/vectors.py src/api/v1/vector_similarity_models.py src/api/v1/vectors_similarity_router.py tests/unit/test_vectors_similarity_router.py tests/unit/test_vectors_batch_similarity_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vectors_similarity_router.py tests/unit/test_vectors_batch_similarity_delegation.py tests/unit/test_vector_batch_similarity_pipeline.py tests/unit/test_batch_similarity.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_batch_similarity_edges.py tests/unit/test_batch_similarity_empty_and_cap.py tests/unit/test_batch_similarity_faiss_unavailable.py tests/unit/test_faiss_degraded_batch.py`
- `.venv311/bin/python -m pytest -q tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py tests/unit/test_api_route_uniqueness.py tests/unit/test_vectors_crud_router.py tests/unit/test_vectors_list_router.py tests/unit/test_vectors_migration_read_router.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_batch_similarity_empty_results.py tests/unit/test_vectors_similarity_router.py tests/unit/test_vectors_batch_similarity_delegation.py tests/unit/test_vector_batch_similarity_pipeline.py tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py`
- `git diff --check`

## 本地验证结果

- `py_compile`：通过。
- `flake8`：通过。
- Batch similarity route + delegation + pipeline + main API behavior：`18 passed, 7 warnings`。
- Batch similarity edge cases + empty/cap + Faiss degraded/fallback：`15 passed, 1 skipped, 7 warnings`。
- OpenAPI + route uniqueness + existing split router ownership：`8 passed, 7 warnings`。
- Empty result + metrics regression + route/contract subset：`18 passed, 7 warnings`。
- `git diff --check`：通过。
- OpenAPI operationId：`batch_similarity_api_v1_vectors_similarity_batch_post`。
- OpenAPI 200 response `$ref`：`#/components/schemas/BatchSimilarityResponse`。
- OpenAPI component keys：`BatchSimilarityRequest`、`BatchSimilarityResponse`，未生成模块限定 duplicate schema。

## Contract 检查点

- `POST /api/v1/vectors/similarity/batch` route owner：`src.api.v1.vectors_similarity_router`
- Endpoint function name：`batch_similarity`
- OpenAPI operationId：保持 `batch_similarity_api_v1_vectors_similarity_batch_post`
- Response model schema：`BatchSimilarityResponse`
- Response schema ref：保持 `#/components/schemas/BatchSimilarityResponse`
- 兼容 patch 面：继续支持 `src.api.v1.vectors.run_vector_batch_similarity` 等旧 patch path

## 结论

Batch similarity router 拆分已完成本地验证。

本次未移动 shared filter helper 或 core pipeline，避免把 router 拆分扩大为 helper ownership 重构。下一步建议继续拆 `vectors` backend reload admin router，重点保护 admin token dependency、reload metrics 和 failure contract。
