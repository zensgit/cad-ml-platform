# Claude Collaboration Batch 5 Execution Runbook

日期：2026-03-30

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH5_DEVELOPMENT_PLAN_20260330.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH5_EXECUTION_RUNBOOK_20260330.md 执行。
现在只允许做 Batch 5A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH5_VALIDATION_LEDGER_20260330.md，
并明确停止，等待人工验证。
不允许提前做 Batch 5B。
```

---

## 总执行规则

1. 必须先做 Batch 5A
2. Batch 5A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 5B
4. 不允许修改 Development Plan 中写死的接口和范围
5. 不允许新建新的 summary owner / metrics owner
6. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 5A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `scripts/ci/refresh_eval_reporting_stack.py`
   - `scripts/ci/generate_eval_reporting_bundle.py`
   - `scripts/ci/check_eval_reporting_bundle_health.py`
   - `scripts/ci/generate_eval_reporting_index.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新 workflow regression tests
4. 跑验证命令
5. 补齐该批 design/validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 5A 必跑命令

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py -q
```

### Batch 5A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH5_VALIDATION_LEDGER_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_VALIDATION_20260330.md`

---

## Batch 5B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 5A ledger 和 validation MD
2. 实现 thin summary helper
3. 修改 `.github/workflows/evaluation-report.yml`
4. 新增/更新 workflow regression tests
5. 跑验证命令
6. 补齐该批 design/validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 5B 必跑命令

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/summarize_eval_reporting_stack_status.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_summarize_eval_reporting_stack_status.py \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py -q
```

### Batch 5B 合并回归

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_summarize_eval_reporting_stack_status.py \
  tests/unit/test_generate_eval_reporting_bundle.py \
  tests/unit/test_check_eval_reporting_bundle_health.py \
  tests/unit/test_generate_eval_reporting_index.py \
  tests/unit/test_refresh_eval_reporting_stack.py \
  tests/unit/test_generate_eval_report.py \
  tests/unit/test_generate_eval_report_v2.py -q
```

### Batch 5B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH5_VALIDATION_LEDGER_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 本批变更摘要
2. 修改文件列表
3. 新增文件列表
4. design MD 路径
5. validation MD 路径
6. 实际命令
7. 实际结果
8. 未解决风险
9. 一句明确结论：
   - `Batch 5A complete, stopped for validation`
   - 或 `Batch 5B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 5A accepted`

Claude 才能继续 Batch 5B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
