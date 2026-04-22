# Phase 3 Vectors Migration Read Router Extraction Development Plan

日期：2026-04-22

## 目标
- 把 `src/api/v1/vectors.py` 里只读的 migration 路由整体拆到独立 router。
- 继续保留 `src.api.v1.vectors.*` 的 monkeypatch 面，不改现有测试对模块级 helper 的 patch 习惯。

## 变更范围
- 新增子路由：`src/api/v1/vectors_migration_read_router.py`
- 收口路由：
  - `GET /api/v1/vectors/migrate/preview`
  - `GET /api/v1/vectors/migrate/status`
  - `GET /api/v1/vectors/migrate/summary`
  - `GET /api/v1/vectors/migrate/pending`
  - `GET /api/v1/vectors/migrate/pending/summary`
  - `GET /api/v1/vectors/migrate/plan`
  - `GET /api/v1/vectors/migrate/trends`
- 调整：`src/api/v1/vectors.py`
- 新增测试：`tests/unit/test_vectors_migration_read_router.py`

## 设计约束
- `vectors.py` 里保留 `_collect_vector_migration_pending_candidates(...)`、`_resolve_vector_migration_target_version()` 等兼容 helper。
- 新 router 运行时通过 `src.api.v1.vectors` 模块取 patch 点，避免把 patch 面迁移到新模块后打断旧测试。
- 不触碰 `/api/v1/vectors/migrate` 和 `/api/v1/vectors/migrate/pending/run` 这两条写路径。

## 风险控制
- 不改 request/response schema。
- 不改 migration history、scan limit、qdrant readiness 的既有语义。
- 增加 route ownership smoke test，锁住新旧模块边界。
