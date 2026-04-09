# Active Learning Evidence Review Queue Validation 2026-03-08

## 目标

让 active-learning 样本不只暴露 `score_breakdown` 原始 JSON，还直接给 reviewer
可读的 evidence 结构，减少人工复核时的解析成本。

## 改动

- 扩展 [active_learning.py](/private/tmp/cad-ml-platform-active-learning-evidence-20260308/src/core/active_learning.py)
- `ActiveLearningSample` 新增：
  - `evidence_count`
  - `evidence_sources`
  - `evidence_summary`
  - `evidence`
- evidence 统一从 `score_breakdown` 自动派生：
  - `source_contributions`
  - `hybrid_explanation.summary`
  - `hybrid_rejection`
  - `decision_path`
  - `fusion_metadata`
- 覆盖面：
  - pending API
  - review queue API
  - review queue export
  - training export
  - file-store 旧样本重载

## 验证

执行：

```bash
python3 -m py_compile src/core/active_learning.py tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py tests/unit/test_active_learning_loop.py
flake8 src/core/active_learning.py tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py tests/unit/test_active_learning_loop.py --max-line-length=100
pytest -q tests/test_active_learning_api.py tests/unit/test_active_learning_export_context.py tests/unit/test_active_learning_loop.py
```

## 结果预期

- pending/review-queue 响应直接带 evidence 字段
- review queue CSV / JSONL 带 evidence 相关列
- training export 保留 evidence，便于后续人工复核和再训练分析
