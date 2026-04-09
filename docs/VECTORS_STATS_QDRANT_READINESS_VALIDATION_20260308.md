# Vectors Stats Qdrant Readiness Validation 2026-03-08

## 目标

为 `GET /api/v1/vectors_stats/stats` 和 `GET /api/v1/vectors_stats/distribution`
补充 Qdrant 原生观测字段，让 benchmark / 运维侧可以直接看到：

- collection 状态
- points / indexed / unindexed 数量
- indexed ratio
- scan 是否被 `VECTOR_STATS_SCAN_LIMIT` 截断
- readiness 与 readiness hints

这条线只做只读观测增强，不重复已有 migration / pending / plan 能力。

## 关键改动

- `src/api/v1/vectors_stats.py`
  - 新增 `backend_health`
  - Qdrant 下补充：
    - `collection_name`
    - `collection_status`
    - `points_count`
    - `indexed_vectors_count`
    - `unindexed_vectors_count`
    - `indexed_ratio`
    - `observed_vectors_count`
    - `scan_limit`
    - `scan_truncated`
    - `vector_size`
    - `distance`
    - `readiness`
    - `readiness_hints`
- `tests/unit/test_vector_stats.py`
  - 新增 Qdrant readiness 断言
- `tests/unit/test_vector_distribution_endpoint.py`
  - 新增 scan truncation / partial scan readiness 断言

## 设计取舍

- `backend_health` 只在 `VECTOR_STORE_BACKEND=qdrant` 时返回，避免污染 memory/redis 响应。
- `readiness` 使用轻量规则推导：
  - `empty`
  - `degraded`
  - `partial_scan`
  - `indexing`
  - `ready`
- `readiness_hints` 明确告诉调用侧为什么不是 `ready`，而不是只给一个状态值。

## 验证命令

```bash
python3 -m py_compile src/api/v1/vectors_stats.py \
  tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py

flake8 src/api/v1/vectors_stats.py \
  tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py \
  --max-line-length=100

pytest -q tests/unit/test_vector_stats.py \
  tests/unit/test_vector_distribution_endpoint.py
```

## 验证结果

- `py_compile` 通过
- `flake8` 通过
- `pytest` 通过：`5 passed`

## 结果

`vectors_stats` 现在可以直接回答：

- Qdrant 当前是否 ready
- 索引是否仍在回填
- 统计结果是不是 partial scan
- 当前 summary 是否足以用于 benchmark / 运维判断
