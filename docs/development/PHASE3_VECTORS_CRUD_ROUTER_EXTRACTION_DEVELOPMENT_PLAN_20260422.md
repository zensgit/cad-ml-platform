# Phase 3 Vectors CRUD Router Extraction Development Plan

日期：2026-04-22

## 目标
- 把 `src/api/v1/vectors.py` 里 live CRUD 路由拆到独立 router。
- 继续保留 `src.api.v1.vectors.*` 的 monkeypatch 面，不改现有测试 patch 习惯。

## 变更范围
- 新增共享模型：`src/api/v1/vector_crud_models.py`
- 新增子路由：`src/api/v1/vectors_crud_router.py`
- 收口路由：
  - `POST /api/v1/vectors/delete`
  - `POST /api/v1/vectors/register`
  - `POST /api/v1/vectors/search`
- 调整：`src/api/v1/vectors.py`
- 新增测试：`tests/unit/test_vectors_crud_router.py`

## 设计约束
- `vectors.py` 里继续保留 `run_vector_delete_pipeline`、`run_vector_register_pipeline`、`run_vector_search_pipeline`、`_get_qdrant_store_or_none` 等 patch 面。
- 新 router 运行时通过 `src.api.v1.vectors` 模块取 patch 点，避免把 patch 面迁移到新模块后打断旧测试。
- 不触碰 `list/similarity-batch/backend-reload` 与 migration 路由。

## 风险控制
- 不改 request/response schema 语义，只把 CRUD 模型从 `vectors.py` 提取到共享模型模块。
- 不改 qdrant/memory 分支行为。
- 增加 route ownership smoke test，锁住新旧模块边界。
