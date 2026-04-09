# Qdrant Native Filtered Queries Validation

Date: 2026-03-07

## Goal

将 coarse/fine/decision_source/is_coarse_label 过滤从 API 层后过滤，下沉到
Qdrant 原生 query/filter/scroll 路径，覆盖：

- `GET /api/v1/vectors`
- `POST /api/v1/vectors/search`
- `POST /api/v1/analyze/similarity/topk`

同时保留 memory/redis/FAISS 现有行为作为回退路径。

## Key Changes

### Qdrant store

Updated:

- `src/core/vector_stores/qdrant_store.py`

Added:

- `_build_query_filter()`
- `_to_result()`
- `list_vectors()`

Adjusted:

- `search_similar()` 改为复用统一 filter builder
- `delete_vectors_by_filter()` / `count()` 改为复用统一 filter builder

### API routing

Updated:

- `src/api/v1/vectors.py`
- `src/api/v1/analyze.py`

Behavior:

- `VECTOR_STORE_BACKEND=qdrant` 时：
  - `/api/v1/vectors?source=qdrant` 直接走 Qdrant `count + scroll`
  - `/api/v1/vectors/search` 直接走 Qdrant `search`
  - `/api/v1/analyze/similarity/topk` 直接走 Qdrant `get_vector + search`
- 若 Qdrant 不可用，自动回退到现有 memory/redis/FAISS 路径

### Tests

Updated:

- `tests/unit/test_qdrant_vector_store.py`
- `tests/unit/test_vectors_module_endpoints.py`
- `tests/unit/test_similarity_topk.py`

Added coverage:

- Qdrant list with coarse filter
- Qdrant native vector search with coarse filter
- Qdrant native similarity topk with coarse filter
- Qdrant scroll/count filter wiring

## Validation Commands

```bash
python3 -m py_compile \
  src/core/vector_stores/qdrant_store.py \
  src/api/v1/vectors.py \
  src/api/v1/analyze.py \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py

flake8 \
  src/core/vector_stores/qdrant_store.py \
  src/api/v1/vectors.py \
  src/api/v1/analyze.py \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_qdrant_vector_store.py \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py

pytest -q tests/contract/test_openapi_schema_snapshot.py
```

## Results

- `py_compile`: pass
- `flake8`: pass
- targeted `pytest`: `23 passed, 19 skipped`
- openapi snapshot: `1 passed`

## Notes

- 当前实现已经把 coarse contract 真正下沉到了 Qdrant 原生过滤面。
- `GET /api/v1/vectors` 的数值分页仍基于 `count + scroll(limit=offset+limit)`，
  不是游标分页；这次优先解决的是“服务端过滤”和“主链接线”。
- 现有 API 契约未被破坏；新增的是 `source=qdrant` 路径和后台原生实现。
