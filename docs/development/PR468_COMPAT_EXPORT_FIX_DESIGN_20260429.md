# PR468 Compatibility Export Fix Design

日期：2026-04-29

## 背景

PR #468 的 OpenAPI fast gate 已修复，但远端 `unit-tier` / `tests (3.10)` / `tests (3.11)` 在全量 unit collection 阶段失败。

失败根因不是 CRUD router 行为，而是前序 `vectors.py` router 拆分后，旧测试与旧调用方仍从 `src.api.v1.vectors` 直接导入迁移相关模型和 route handler：

- `VectorMigrationPreviewResponse`
- `VectorMigrationTrendsResponse`
- `migrate_vectors`
- `preview_migration`

这些对象已经移动到 split router 或 shared model module，但 `vectors.py` 没有保留兼容 facade。

## 设计原则

- 不回滚 router 拆分。
- 不复制 route handler 实现。
- 不改变 OpenAPI path、response model 或 operationId。
- `src.api.v1.vectors` 继续作为兼容导出面，旧 import 不破坏。
- 新 split routers 继续拥有实际 route handler。

## 变更设计

### 1. 恢复 `vectors.py` 兼容 re-export

`src/api/v1/vectors.py` 从以下模块导入并导出迁移相关对象：

- `src.api.v1.vector_migration_models`
- `src.api.v1.vectors_migration_read_router`
- `src.api.v1.vectors_write_router`

这样旧代码仍可执行：

```python
from src.api.v1.vectors import VectorMigrationPreviewResponse, migrate_vectors
```

### 2. 显式维护 `__all__`

`vectors.py` 的 `__all__` 显式列出兼容 facade 对象，避免 flake8 将 re-export import 判定为未使用。

### 3. 修复时间敏感测试

`tests/unit/test_vector_migration_trends_pipeline.py` 原本使用固定日期 `2026-04-21T10:00:00` 作为 24 小时窗口内样本。随着当前日期推进，该样本会落到窗口外。

改为：

```python
datetime.utcnow() - timedelta(minutes=5)
```

测试语义保持为“窗口内迁移记录应计入 trends”，但不再依赖固定日期。

## 影响范围

- 运行时 route 行为不变。
- OpenAPI contract 不变。
- 旧 import 面恢复。
- 全量 unit collection 不再因迁移对象缺失中断。

