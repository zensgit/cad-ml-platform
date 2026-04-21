# Phase 3 Vector Migration Trends Pipeline Extraction Development Plan

日期：2026-04-21

## 目标
- 抽取 `src/api/v1/vectors.py` 中 `/migrate/trends` 的 history 过滤、统计汇总和版本分布读取逻辑。
- 保持 `migrate_trends()` 的函数名、route 签名以及 `_get_qdrant_store_or_none` patch 面不变。

## 变更范围
- 新增共享 helper：`src/core/vector_migration_trends_pipeline.py`
- 收口路由：
  - `GET /api/v1/vectors/migrate/trends`
- 新增测试：
  - `tests/unit/test_vector_migration_trends_pipeline.py`
  - `tests/unit/test_vectors_migration_trends_delegation.py`

## 设计约束
- route 仍返回 `VectorMigrationTrendsResponse`
- helper 复用 `collect_vector_migration_distribution_snapshot()`，避免再次复制 qdrant/memory 分布读取
- `migrate_trends()` 继续作为直接导入调用入口，兼容现有 async 函数级测试

## 风险控制
- 不触碰 `/migrate` 与 `/migrate/preview`、`/migrate/plan` 的现有逻辑
- time window 过滤保留“无效时间戳也纳入统计”的旧语义
- 用 route delegation test 锁住薄封装边界
