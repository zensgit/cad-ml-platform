# Qdrant Observability Summary Validation 2026-03-08

## 目标

在 `vectors_stats` 已有 readiness 基础上，再补一层更稳的只读观测实现：

- 在 `QdrantVectorStore` 中增加不创建 collection 的 `inspect_collection()`
- 让 `vectors_stats` 优先复用这个只读快照
- 给 `GET /api/v1/maintenance/stats` 增加精简版 Qdrant 摘要

## 关键改动

- `src/core/vector_stores/qdrant_store.py`
  - 新增 `inspect_collection()`
  - 输出：
    - `reachable`
    - `collection_exists`
    - `collection_status`
    - `points_count`
    - `vectors_count`
    - `indexed_vectors_count`
    - `unindexed_vectors_count`
    - `indexing_progress`
    - `requested_config`
    - `error`
- `src/api/v1/vectors_stats.py`
  - `backend_health` 改为优先读取只读 inspect 快照
  - 新增：
    - `reachable`
    - `collection_exists`
    - `on_disk`
    - `timeout_seconds`
    - `error`
    - 更准确的 readiness 分类：`unavailable` / `missing_collection`
- `src/api/v1/maintenance.py`
  - `vector_store.qdrant` 新增精简摘要：
    - `reachable`
    - `collection_exists`
    - `collection_status`
    - `unindexed_vectors_count`
    - `indexing_progress`
    - `error`

## 验证命令

```bash
python3 -m py_compile \
  src/core/vector_stores/qdrant_store.py \
  src/api/v1/vectors_stats.py \
  src/api/v1/maintenance.py \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_maintenance_endpoint_coverage.py

flake8 \
  src/core/vector_stores/qdrant_store.py \
  src/api/v1/vectors_stats.py \
  src/api/v1/maintenance.py \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_maintenance_endpoint_coverage.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  tests/unit/test_maintenance_endpoint_coverage.py
```

## 结果

- `inspect_collection()` 不会触发 collection 创建
- `vectors_stats` 和 `maintenance/stats` 现在共享同一套 Qdrant 只读观测语义
- 运维侧可以直接看到索引回填进度和 collection 是否缺失，而不是只看到总量
