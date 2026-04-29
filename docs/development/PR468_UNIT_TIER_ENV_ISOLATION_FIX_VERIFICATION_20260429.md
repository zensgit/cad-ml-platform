# PR468 Unit-Tier Env Isolation Fix Verification

日期：2026-04-29

## 远端失败摘要

GitHub Actions `unit-tier` 失败摘要：

- Job：`73509260885`
- Run：`25088501803`
- 命令：`bash scripts/test_with_local_api.sh --suite unit`
- 失败结果：`2 failed, 9904 passed, 117 skipped, 26 warnings`

失败测试：

- `tests/unit/test_qdrant_store_helper.py::test_get_qdrant_store_or_none_returns_store_when_qdrant_backend_enabled`
- `tests/unit/test_vector_backend_reload_pipeline.py::test_vector_backend_reload_pipeline_success_sets_backend_env`

## 本地验证命令

- `.venv311/bin/flake8 tests/unit/test_backend_reload_failures.py tests/unit/test_qdrant_store_helper.py tests/unit/test_vector_backend_reload_pipeline.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_backend_reload_failures.py tests/unit/test_qdrant_store_helper.py tests/unit/test_vector_backend_reload_pipeline.py`
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_pr468_pycache .venv311/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_backend_reload_failures.py::test_backend_reload_concurrent_conflict tests/unit/test_qdrant_store_helper.py::test_get_qdrant_store_or_none_returns_store_when_qdrant_backend_enabled tests/unit/test_vector_backend_reload_pipeline.py::test_vector_backend_reload_pipeline_success_sets_backend_env --tb=short`
- `bash scripts/test_with_local_api.sh --suite unit`

## 本地验证结果

- `flake8`：通过。
- 三个相关测试文件：`19 passed, 7 warnings`。
- 远端失败顺序窄回归：`3 passed, 7 warnings`。
- 完整 unit-tier 本地等价验证：`9939 passed, 116 skipped, 26 warnings in 155.12s`。

## 结论

`unit-tier` 的两个失败来自测试并发污染：

- 修复前：并发线程内 `patch("os.getenv")` 可能泄漏 mock，后续 backend env 测试读到错误 backend。
- 修复后：环境变量由 `monkeypatch.setenv` 统一设置和恢复，线程内不再 patch 进程全局 `os.getenv`。

本地完整 `unit-tier` 已通过，PR #468 可以重跑远端 checks 验证。

