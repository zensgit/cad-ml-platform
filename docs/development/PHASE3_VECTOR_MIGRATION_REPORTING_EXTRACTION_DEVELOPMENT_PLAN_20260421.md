# Phase 3 Vector Migration Reporting Extraction Development Plan

日期：2026-04-21

## 目标
- 抽取 `src/api/v1/vectors.py` 中 `/migrate/status` 与 `/migrate/summary` 的重复分布读取与 history 聚合逻辑。
- 保持现有路由签名、`_get_qdrant_store_or_none` patch 面、response model 与返回契约不变。

## 变更范围
- 新增共享 helper：`src/core/vector_migration_reporting_pipeline.py`
- 收口路由：
  - `GET /api/v1/vectors/migrate/status`
  - `GET /api/v1/vectors/migrate/summary`
- 新增测试：
  - `tests/unit/test_vector_migration_reporting_pipeline.py`
  - `tests/unit/test_vectors_migration_reporting_delegation.py`

## 设计约束
- `src/core` 不反向依赖 `api/v1` schema，helper 仅返回 `dict`
- `vectors.py` 继续负责 `VectorMigrationStatusResponse(**payload)` / `VectorMigrationSummaryResponse(**payload)` 包装
- qdrant 与 memory 分布读取语义保持一致

## 风险控制
- 不触碰 `/migrate` 主迁移写路径
- 不修改 readiness 计算规则
- 增加 route delegation smoke test，防止重复逻辑回流到 `vectors.py`
