# Qdrant Maintenance Error Taxonomy Validation 2026-03-08

## 目标

让 `GET /api/v1/maintenance/stats` 在 `VECTOR_STORE_BACKEND=qdrant` 时输出可操作的
错误 taxonomy，而不是只返回一段原始错误文本。

## 实现

- 扩展 [maintenance.py](/private/tmp/cad-ml-platform-main-post-168/src/api/v1/maintenance.py)
- 新增：
  - `error_type`
  - `error_severity`
  - `error_hint`
- 当前分类覆盖：
  - `none`
  - `timeout`
  - `connection`
  - `authentication`
  - `not_found`
  - `sdk_unavailable`
  - `configuration`
  - `unknown`

## 验证

执行：

```bash
python3 -m py_compile src/api/v1/maintenance.py tests/unit/test_maintenance_endpoint_coverage.py
flake8 src/api/v1/maintenance.py tests/unit/test_maintenance_endpoint_coverage.py --max-line-length=100
pytest -q tests/unit/test_maintenance_endpoint_coverage.py -k "get_stats_qdrant_backend"
```

预期：

- 正常 Qdrant observability 返回：
  - `error_type=none`
  - `error_severity=none`
  - `error_hint=null`
- 超时类错误返回：
  - `error_type=timeout`
  - `error_severity=critical`
  - `error_hint` 含超时排障建议

## 价值

这条线补的是 benchmark 超越目标里的“向量底座可运维性”：

- 不只知道 Qdrant 出错
- 还能快速知道错误类别和优先级
- 方便后续接 benchmark scorecard / maintenance dashboard / runbook
