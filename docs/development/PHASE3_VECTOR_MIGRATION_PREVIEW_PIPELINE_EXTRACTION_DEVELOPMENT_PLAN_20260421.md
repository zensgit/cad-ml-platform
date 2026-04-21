# Phase 3 Vector Migration Preview Pipeline Extraction Development Plan

日期：2026-04-21

## 目标
- 抽取 `src/api/v1/vectors.py` 中 `/migrate/preview` 的主体验证与预览推导逻辑。
- 保持 `preview_migration()` 的函数名、route 签名以及 `_get_qdrant_store_or_none` / `_collect_qdrant_preview_samples` / `_prepare_vector_for_upgrade` 这些 patch 面不变。

## 变更范围
- 新增共享 helper：`src/core/vector_migration_preview_pipeline.py`
- 收口路由：
  - `GET /api/v1/vectors/migrate/preview`
- 新增测试：
  - `tests/unit/test_vector_migration_preview_pipeline.py`
  - `tests/unit/test_vectors_migration_preview_delegation.py`

## 设计约束
- route 仍返回 `VectorMigrationPreviewResponse`
- helper 负责版本校验、qdrant/memory 取样、维度变化统计、warning 推导
- `preview_migration()` 继续作为直接导入调用入口，避免现有 async 函数级测试失效

## 风险控制
- 不触碰 `/migrate` 写路径
- 不改 `_collect_qdrant_preview_samples` 与 `_prepare_vector_for_upgrade` 语义
- 用 route delegation test 锁住薄封装边界
