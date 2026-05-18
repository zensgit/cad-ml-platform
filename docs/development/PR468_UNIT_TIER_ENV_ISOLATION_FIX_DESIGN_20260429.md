# PR468 Unit-Tier Env Isolation Fix Design

日期：2026-04-29

## 背景

PR #468 在恢复 migration compatibility exports 后，远端 `unit-tier` 继续失败，但失败点已经从 collection/import error 变成两个运行期断言：

- `tests/unit/test_qdrant_store_helper.py::test_get_qdrant_store_or_none_returns_store_when_qdrant_backend_enabled`
- `tests/unit/test_vector_backend_reload_pipeline.py::test_vector_backend_reload_pipeline_success_sets_backend_env`

两个失败都表现为后续测试期望 `VECTOR_STORE_BACKEND=qdrant` 或 `faiss`，但运行时仍读到 `memory`。

## 根因

`tests/unit/test_backend_reload_failures.py::test_backend_reload_concurrent_conflict` 在两个并发线程里使用：

```python
patch("os.getenv")
```

`patch("os.getenv")` 是进程级全局替换，不是线程局部替换。两个线程的 patch context 交叠后，恢复顺序可能错乱，导致 `os.getenv` 在测试结束后仍保持为 mock，并继续返回 `memory`。

这会污染后续测试，即使后续测试使用 `patch.dict` 或 `monkeypatch.setenv` 设置了 backend env，调用 `os.getenv()` 时仍可能命中泄漏的 mock。

## 设计原则

- 不修改产品代码，因为失败来自测试隔离问题。
- 不降低并发测试覆盖，仍保留两个线程同时请求 backend reload endpoint。
- 不在并发线程内 patch 进程全局函数。
- 使用 pytest `monkeypatch` 管理环境变量生命周期，保证测试结束后自动恢复。

## 变更设计

### 1. 改用 `monkeypatch.setenv`

并发测试入口增加 `monkeypatch` fixture：

```python
def test_backend_reload_concurrent_conflict(client, monkeypatch):
```

在线程启动前统一设置：

```python
monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")
```

这样测试仍以 memory backend 为当前 backend，但不需要 patch `os.getenv`。

### 2. 线程内只保留目标行为 patch

每个线程仍 patch：

```python
src.core.similarity.reload_vector_store_backend
```

该 patch 只模拟 slow reload，用于保留并发冲突覆盖；不再 patch `os.getenv`。

## 影响范围

- 产品行为无变化。
- backend reload 并发测试语义不变。
- 后续 qdrant/faiss backend 测试不再受 `os.getenv` mock 污染。
- PR #468 的 `unit-tier` 失败路径被纳入本地复现和回归验证。

