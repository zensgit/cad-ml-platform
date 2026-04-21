# Phase 3 Vector Migration Plan Pipeline Extraction Development Plan

日期：2026-04-21

## 目标
- 抽取 `src/api/v1/vectors.py` 中 `/migrate/pending/summary` 与 `/migrate/plan` 的重复推导逻辑。
- 保持现有路由签名、`_collect_vector_migration_pending_candidates` 与 `_resolve_vector_migration_target_version` patch 面不变。

## 变更范围
- 新增共享 helper：`src/core/vector_migration_plan_pipeline.py`
- 收口路由：
  - `GET /api/v1/vectors/migrate/pending/summary`
  - `GET /api/v1/vectors/migrate/plan`
- 新增测试：
  - `tests/unit/test_vector_migration_plan_pipeline.py`
  - `tests/unit/test_vectors_migration_plan_delegation.py`

## 设计约束
- `src/core` 只返回 payload `dict`，不直接依赖 `api/v1` response model
- `vectors.py` 继续负责 `VectorMigrationPendingSummaryResponse(**payload)` 与 `VectorMigrationPlanResponse(**payload)` 包装
- 批次排序、partial-scan 约束、推荐请求 payload 语义保持不变

## 风险控制
- 不触碰 `/migrate` 与 `/migrate/pending/run` 写路径
- 不改 pending candidates 收集逻辑
- 增加 route delegation 测试，防止后续重复逻辑回流
