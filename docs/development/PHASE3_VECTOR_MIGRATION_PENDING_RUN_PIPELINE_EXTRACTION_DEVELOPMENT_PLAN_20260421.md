# Phase 3 Vector Migration Pending Run Pipeline Extraction Development Plan

日期：2026-04-21

## 目标
- 抽取 `src/api/v1/vectors.py` 中 `/migrate/pending/run` 的约束判断与迁移请求构造逻辑。
- 保持 `migrate_pending_run()` 的函数名、route 签名以及 `_collect_vector_migration_pending_candidates` / `_resolve_vector_migration_target_version` / `migrate_vectors` 这些 patch 面不变。

## 变更范围
- 新增共享 helper：`src/core/vector_migration_pending_run_pipeline.py`
- 收口路由：
  - `POST /api/v1/vectors/migrate/pending/run`
- 新增测试：
  - `tests/unit/test_vector_migration_pending_run_pipeline.py`
  - `tests/unit/test_vectors_migration_pending_run_delegation.py`

## 设计约束
- route 继续返回 `VectorMigrateResponse`
- helper 只处理：
  - target version 解析
  - pending candidates 查询
  - qdrant partial scan 约束判断
  - 迁移请求对象构造与 `migrate_vectors()` 调用
- 不触碰实际迁移写路径实现

## 风险控制
- 不修改 `pending candidates` 收集逻辑
- 不修改 `migrate_vectors()` 语义
- 用 route delegation test 锁住薄封装边界
