# Phase 3 Vectors Batch Similarity Router Design

日期：2026-04-29

## 背景

Phase 3 router 收口在 #471 完成 `GET /api/v1/vectors/` list router 拆分后，下一项是拆分：

```text
POST /api/v1/vectors/similarity/batch
```

当前依赖链仍是 stacked PR：

- #468：`phase3-vectors-crud-router-20260422`
- #471：`phase3-vectors-list-router-20260429`
- 本次分支：`phase3-vectors-batch-similarity-router-20260429`

这样可以继续保持小 PR，不把 backend reload admin route 的 auth/metrics 风险混入 batch similarity route。

## 目标

- 将 batch similarity endpoint 从 `src/api/v1/vectors.py` 拆到独立 router。
- 将 batch similarity request/response models 拆到独立 model module。
- 保持 OpenAPI path、operationId、response schema 不变。
- 保持 `src.api.v1.vectors.*` monkeypatch 兼容面不变。
- 不移动 `run_vector_batch_similarity` pipeline 或共享 filter helper。

## 设计

### 1. 新增 batch similarity models

新增：

- `src/api/v1/vector_similarity_models.py`

包含：

- `BatchSimilarityRequest`
- `BatchSimilarityItem`
- `BatchSimilarityResponse`

`src/api/v1/vectors.py` 继续 re-export 这些模型，保留旧导入兼容。

### 2. 新增 batch similarity router

新增：

- `src/api/v1/vectors_similarity_router.py`

该 router 持有：

```text
POST /similarity/batch
```

聚合后最终 API path 仍为：

```text
POST /api/v1/vectors/similarity/batch
```

endpoint function 名称保持为：

```python
batch_similarity
```

因此 OpenAPI operationId 保持：

```text
batch_similarity_api_v1_vectors_similarity_batch_post
```

### 3. 保留兼容 patch facade

Router 内部不直接固定引用 pipeline 和 helper，而是在 endpoint 内延迟读取：

```python
from src.api.v1 import vectors as vectors_module
```

继续通过 `vectors_module.*` 访问：

- `run_vector_batch_similarity`
- `ErrorCode`
- `build_error`
- `_get_qdrant_store_or_none`
- `_build_vector_filter_conditions`

这样现有测试仍可 patch：

```python
monkeypatch.setattr("src.api.v1.vectors.run_vector_batch_similarity", ...)
```

### 4. 暂不移动 helper ownership

`_build_vector_filter_conditions` 同时被 list/search/batch similarity 使用，本次不移动。

原因：

- 当前目标是 router ownership 收口，不是 helper ownership 重构。
- 过早移动 helper 会扩大 monkeypatch 兼容面风险。
- backend reload router 尚未拆出，`vectors.py` 仍需要作为兼容 facade 存在。

## 风险控制

- 新增 route ownership guard，确保 batch route 归属 `src.api.v1.vectors_similarity_router`。
- 运行 batch similarity 主路径、边界、Faiss degraded/fallback 回归。
- 运行 OpenAPI operationId、schema snapshot、route uniqueness 回归。
- 直接检查 OpenAPI 200 response `$ref`，防止 model 移动引入模块限定 schema key。

