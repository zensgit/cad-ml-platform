# Phase 3 Vectors List Router Verification

日期：2026-04-29

## 变更摘要

本次将 `GET /api/v1/vectors/` 从 `src/api/v1/vectors.py` 拆到独立 list router：

- 新增 `src/api/v1/vector_list_models.py`
- 新增 `src/api/v1/vectors_list_router.py`
- 新增 `tests/unit/test_vectors_list_router.py`
- 更新 `src/api/v1/vectors.py` include list router，并继续 re-export list endpoint/model
- 更新 `src/api/v1/analyze_legacy_redirects.py` 复用新 list response model，避免 OpenAPI duplicate schema 命名漂移

## 本地验证命令

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_list_router_pycache python3 -m py_compile src/api/v1/vectors.py src/api/v1/vector_list_models.py src/api/v1/vectors_list_router.py src/api/v1/analyze_legacy_redirects.py`
- `.venv311/bin/flake8 src/api/v1/vectors.py src/api/v1/vector_list_models.py src/api/v1/vectors_list_router.py src/api/v1/analyze_legacy_redirects.py tests/unit/test_vectors_list_router.py tests/unit/test_vectors_list_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vectors_list_router.py tests/unit/test_vectors_list_delegation.py tests/unit/test_vector_list_pipeline.py tests/unit/test_vectors_module_endpoints.py`
- `.venv311/bin/python -m pytest -q tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py tests/unit/test_api_route_uniqueness.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vectors_list_router.py tests/unit/test_vectors_crud_router.py tests/unit/test_vectors_migration_read_router.py tests/unit/test_analyze_legacy_redirect_router.py tests/unit/test_vectors_list_delegation.py tests/unit/test_vector_list_pipeline.py tests/unit/test_vectors_module_endpoints.py tests/unit/test_deprecated_vector_endpoints.py tests/unit/test_deprecated_endpoints_410.py tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py tests/unit/test_api_route_uniqueness.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_management.py tests/unit/test_vectors_module_endpoints.py tests/unit/test_vectors_list_router.py tests/unit/test_vectors_list_delegation.py tests/unit/test_vector_list_pipeline.py tests/contract/test_openapi_operation_ids.py tests/contract/test_openapi_schema_snapshot.py tests/unit/test_api_route_uniqueness.py`
- `git diff --check`

## 本地验证结果

- `py_compile`：通过。
- `flake8`：通过。
- List route + list behavior tests：`20 passed, 7 warnings`。
- OpenAPI/route contract tests：`5 passed, 7 warnings`。
- Expanded route ownership + deprecated redirect + OpenAPI regression：`43 passed, 7 warnings`。
- Vector management + list contract regression：`26 passed, 7 warnings`。
- `git diff --check`：通过。
- OpenAPI operationId：`list_vectors_api_v1_vectors__get`。
- OpenAPI 200 response `$ref`：`#/components/schemas/VectorListResponse`。
- OpenAPI component keys：`VectorListResponse`、`VectorListItem`，未生成模块限定 duplicate schema。

## Contract 检查点

- `GET /api/v1/vectors/` route owner：`src.api.v1.vectors_list_router`
- Endpoint function name：`list_vectors`
- OpenAPI operationId：保持 `list_vectors_api_v1_vectors__get`
- Response model schema：`VectorListResponse`
- Response schema ref：保持 `#/components/schemas/VectorListResponse`
- 兼容 patch 面：继续支持 `src.api.v1.vectors.run_vector_list_pipeline` 等旧 patch path

## 结论

List router 拆分已完成本地验证。

本次未移动 list helper ownership，避免破坏现有 `src.api.v1.vectors.*` monkeypatch 兼容面。后续可在 batch similarity 和 backend reload router 拆分完成后，再单独评估 helper ownership 清理。
