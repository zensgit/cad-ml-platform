# Qdrant Native Mutations Validation

Date: 2026-03-07

## Goal

补齐 Qdrant 主链写路径，让以下接口在 `VECTOR_STORE_BACKEND=qdrant` 时不再依赖
旧的 memory/redis 存储：

- `POST /api/v1/vectors/register`
- `POST /api/v1/vectors/update`
- `POST /api/v1/vectors/delete`

## Key Changes

Updated:

- `src/api/v1/vectors.py`
- `tests/unit/test_vectors_module_endpoints.py`

Behavior:

- `register`
  - Qdrant backend 下直接调用 `qdrant_store.register_vector()`
  - 自动补 `total_dim`
- `delete`
  - Qdrant backend 下先 `get_vector()` 判存在，再 `delete_vector()`
- `update`
  - Qdrant backend 下先 `get_vector()`
  - 复用现有维度校验逻辑
  - 通过 `register_vector()` 重新 upsert 向量与 metadata
  - 保留 `feature_version` 等已有 metadata

Fallback:

- 若未启用或无法连接 Qdrant，仍走现有 memory/redis/FAISS 路径

## Validation Commands

```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vectors_module_endpoints.py

flake8 src/api/v1/vectors.py tests/unit/test_vectors_module_endpoints.py --max-line-length=100

pytest -q \
  tests/unit/test_vectors_module_endpoints.py \
  tests/unit/test_similarity_topk.py \
  tests/unit/test_qdrant_vector_store.py
```

## Results

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `26 passed, 19 skipped`

## Notes

- 这条线和前一层 `Qdrant native filtered queries` 叠加后，Qdrant 在 vector APIs 上已经从
  “只建 metadata index” 提升为：
  - register
  - update
  - delete
  - list
  - search
  - similarity topk

- 这次没有修改 OpenAPI 路由或字段契约，因此未重跑 snapshot 生成，只保留已有契约回归。
