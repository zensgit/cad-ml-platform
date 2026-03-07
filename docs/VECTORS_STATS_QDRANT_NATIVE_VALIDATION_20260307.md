# Vectors Stats Qdrant Native Validation

Date: 2026-03-07

## Goal

让 `vectors_stats` 在 `VECTOR_STORE_BACKEND=qdrant` 时直接使用 Qdrant 原生数据源生成统计与分布结果，而不是继续只依赖 memory/redis。

## Key Changes

Updated:

- `src/api/v1/vectors_stats.py`
- `tests/unit/test_vector_stats.py`
- `tests/unit/test_vector_distribution_endpoint.py`

Behavior:

- `GET /api/v1/vectors_stats/stats`
- `GET /api/v1/vectors_stats/distribution`

在 `backend=qdrant` 时改走 `_summarize_vectors_qdrant()`：

- 通过 `qdrant_store.list_vectors()` 读取样本
- 统计：
  - `by_material`
  - `by_complexity`
  - `by_format`
  - `by_coarse_part_type`
  - `by_decision_source`
  - `versions`
  - `average_dimension`
  - `dominant_coarse_ratio`

## Validation Commands

```bash
python3 -m py_compile src/api/v1/vectors_stats.py tests/unit/test_vector_stats.py tests/unit/test_vector_distribution_endpoint.py

flake8 src/api/v1/vectors_stats.py tests/unit/test_vector_stats.py tests/unit/test_vector_distribution_endpoint.py --max-line-length=100

pytest -q tests/unit/test_vector_stats.py tests/unit/test_vector_distribution_endpoint.py
```

## Results

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `5 passed`

## Notes

- 该实现与现有 redis 路径保持同样的 `scan_limit` 风格，只是数据源换成 Qdrant。
- 这是统计面补齐，未引入新的 API 路由或字段契约破坏。
