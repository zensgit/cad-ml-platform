# Phase 3 Vectors Write Router Extraction Development Plan

日期：2026-04-22

## 目标
- 把 `src/api/v1/vectors.py` 里写路径路由拆到独立 router。
- 继续保留 `src.api.v1.vectors.*` 的 monkeypatch 面，不改现有测试 patch 习惯。

## 变更范围
- 新增子路由：`src/api/v1/vectors_write_router.py`
- 收口路由：
  - `POST /api/v1/vectors/update`
  - `POST /api/v1/vectors/migrate`
  - `POST /api/v1/vectors/migrate/pending/run`
- 调整：`src/api/v1/vectors.py`
- 新增测试：`tests/unit/test_vectors_write_router.py`

## 设计约束
- `vectors.py` 里保留 `_collect_vector_migration_pending_candidates(...)`、`_resolve_vector_migration_target_version()`、`run_vector_migrate_pipeline`、`run_vector_migration_pending_run_pipeline`、`run_vector_update_pipeline` 等兼容 patch 面。
- 新 router 运行时通过 `src.api.v1.vectors` 模块取 patch 点，避免把 patch 面迁移到新模块后打断旧测试。
- 不改 `/api/v1/vectors/migrate` 写路径的请求/响应模型和历史记录语义。

## 风险控制
- 不触碰 migration read router 已拆出的只读路径。
- 不改 `pending/run -> migrate_vectors` 的调用链，只移动路由承载位置。
- 增加 route ownership smoke test，锁住新旧模块边界。
