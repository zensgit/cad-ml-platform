# PR468 OpenAPI Contract Fix Verification

日期：2026-04-24

## 验证命令

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/api/v1/vectors.py src/api/v1/vectors_crud_router.py src/api/v1/vector_crud_models.py tests/unit/test_vectors_crud_router.py`
- `.venv311/bin/flake8 src/api/v1/vectors.py src/api/v1/vectors_crud_router.py src/api/v1/vector_crud_models.py tests/unit/test_vectors_crud_router.py`
- `.venv311/bin/python -m pytest -q tests/contract/test_openapi_schema_snapshot.py tests/contract/test_openapi_operation_ids.py tests/unit/test_api_route_uniqueness.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vectors_crud_router.py tests/unit/test_vectors_delete_delegation.py tests/unit/test_vectors_register_delegation.py tests/unit/test_vectors_search_delegation.py tests/unit/test_vector_delete_not_found.py tests/unit/test_vectors_module_endpoints.py`
- `make validate-core-fast`

## 结果

- `py_compile` 通过
- `flake8` 通过
- OpenAPI/route contract：`5 passed, 7 warnings`
- Vectors CRUD targeted regression：`22 passed, 7 warnings`
- `make validate-core-fast` 通过

## 关键结论

- `/api/v1/vectors/register` operationId 已恢复为 `register_vector_endpoint_api_v1_vectors_register_post`。
- `config/openapi_schema_snapshot.json` 无需更新，也未被修改。
- 远端 `openapi-fast` 失败对应的本地复现点已修复。

