# PR468 OpenAPI Contract Fix Development

日期：2026-04-24

## 背景

PR #468 将 `/api/v1/vectors` 的 `delete / register / search` CRUD 路由拆分到 `src/api/v1/vectors_crud_router.py`。

远端 `core-fast-gate` 与 `openapi-fast` 失败后，本地复现定位到 OpenAPI snapshot mismatch。根因不是 schema 需要刷新，而是拆分时 `register` 路由处理函数从 `register_vector_endpoint` 改名为 `register_vector`，导致 FastAPI 自动生成的 operationId 从：

- `register_vector_endpoint_api_v1_vectors_register_post`

漂移为：

- `register_vector_api_v1_vectors_register_post`

这属于外部 OpenAPI 契约漂移，不应通过刷新 `config/openapi_schema_snapshot.json` 掩盖。

## 修复

- 将 `src/api/v1/vectors_crud_router.py` 中 `/register` 的处理函数名恢复为 `register_vector_endpoint`。
- 保持路由拆分后的实现边界不变，继续委托 `src.api.v1.vectors.run_vector_register_pipeline`。
- 不修改 OpenAPI snapshot，确保快照测试继续约束外部 API 契约。

## 影响范围

- 运行时行为不变。
- OpenAPI operationId 恢复到拆分前值。
- CRUD router extraction 的内部模块边界保留。

