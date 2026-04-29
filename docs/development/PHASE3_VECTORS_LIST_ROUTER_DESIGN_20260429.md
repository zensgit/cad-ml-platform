# Phase 3 Vectors List Router Design

日期：2026-04-29

## 背景

Phase 3 router 收口的下一项是拆分 `GET /api/v1/vectors/` list endpoint。

当前 PR #468 已通过远端 checks，但仍被 GitHub review gate 阻断：

- `mergeStateStatus=BLOCKED`
- `reviewDecision=REVIEW_REQUIRED`

因此本次开发不继续扩大 #468，而是基于 #468 当前 HEAD 创建 stacked branch：

- `phase3-vectors-list-router-20260429`

后续 #468 合并后，该分支可 rebase 或 retarget 到 `main`。

## 目标

- 将 `GET /api/v1/vectors/` 从聚合文件 `src/api/v1/vectors.py` 拆到独立 router。
- 保持 OpenAPI path、response model、operationId 不变。
- 保持 `src.api.v1.vectors.*` monkeypatch 兼容面不变。
- 不在本次移动 list helper，避免扩大行为风险。

## 设计

### 1. 新增 list models

新增：

- `src/api/v1/vector_list_models.py`

包含：

- `VectorListItem`
- `VectorListResponse`

`src/api/v1/vectors.py` 继续 re-export 这两个模型，保持旧导入兼容。

Deprecated analyze redirect 也复用新的 `VectorListResponse`，避免 OpenAPI components 同时出现 legacy model 和新 model 两套 `VectorListResponse`。

### 2. 新增 list router

新增：

- `src/api/v1/vectors_list_router.py`

该 router 持有：

- `GET /`
- endpoint function：`list_vectors`
- response model：`VectorListResponse`

聚合 router 仍由 `src/api/v1/vectors.py` include：

```python
router.include_router(list_router)
```

在 API 注册层，最终路径仍是：

```text
GET /api/v1/vectors/
```

OpenAPI response schema 继续引用：

```text
#/components/schemas/VectorListResponse
```

### 3. 保留兼容 patch facade

List router 不直接固定引用 core pipeline 和 helper，而是在 endpoint 内延迟读取：

```python
from src.api.v1 import vectors as vectors_module
```

继续通过 `vectors_module.*` 访问：

- `run_vector_list_pipeline`
- `ErrorCode`
- `build_error`
- `_get_qdrant_store_or_none`
- `_resolve_list_source`
- `_build_vector_filter_conditions`
- `_list_vectors_redis`
- `_list_vectors_memory`
- `get_client`

这样现有测试和外部代码仍可 patch：

```python
monkeypatch.setattr("src.api.v1.vectors.run_vector_list_pipeline", ...)
```

### 4. 暂不移动 list helpers

以下 helper 本次仍保留在 `vectors.py`：

- `_resolve_list_source`
- `_list_vectors_memory`
- `_list_vectors_redis`
- `_build_vector_filter_conditions`
- `_matches_vector_label_filters`

原因：

- list、search、batch similarity 仍共享 filter helper。
- 当前测试大量依赖 `src.api.v1.vectors.*` patch 面。
- 提前移动 helper 会把 router 拆分扩大成 helper ownership 重构，增加回归风险。

## 风险控制

- 新增 route ownership 测试，确保 list endpoint 已归属 `src.api.v1.vectors_list_router`。
- 跑 OpenAPI operationId 和 schema snapshot 测试，防止 path/operationId/response model 漂移。
- 直接检查 OpenAPI response `$ref`，防止 Pydantic 因 duplicate model names 生成模块限定 component key。
- 跑 memory/redis/qdrant list 行为测试，保护分页、过滤、invalid source 和响应字段。
