# Benchmark Scorecard Qdrant CI Validation 2026-03-08

## 目标

把 benchmark scorecard 新增的 `qdrant_backend` 组件真正接入
`evaluation-report.yml`，让 CI summary 和 PR comment 能直接看到
Qdrant readiness 状态。

## 改动

- 扩展 workflow dispatch 输入：
  - `benchmark_scorecard_qdrant_readiness_summary`
- 扩展 workflow env：
  - `BENCHMARK_SCORECARD_QDRANT_READINESS_JSON`
- `Generate benchmark scorecard (optional)` 现在会透传：
  - `--qdrant-readiness-summary`
- benchmark step outputs 新增：
  - `qdrant_status`
- job summary 新增：
  - `Benchmark Qdrant status`
- PR comment / signal lights 现在会显示：
  - `qdrant=${benchmarkQdrantStatus}`

## 验证

执行：

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
make validate-openapi
```

## 结果预期

- workflow regression 断言通过
- summary 中出现 `Benchmark Qdrant status`
- PR comment 中出现 `benchmarkQdrantStatus`
- benchmark scorecard 现在不只看分类/复核，也覆盖向量底座 readiness
