# Claude Collaboration Batch 10 Execution Runbook

日期：2026-03-31

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH10_DEVELOPMENT_PLAN_20260331.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH10_EXECUTION_RUNBOOK_20260331.md 执行。
现在只允许做 Batch 10A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH10_VALIDATION_LEDGER_20260331.md，
并明确停止，等待人工验证。
不允许提前做 Batch 10B。
```

---

## 总执行规则

1. 必须先做 Batch 10A
2. Batch 10A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 10B
4. 不允许修改 Development Plan 中写死的接口和范围
5. 不允许新建新的 summary owner / metrics owner
6. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 10A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `scripts/ci/generate_eval_reporting_release_summary.py`
   - `scripts/ci/generate_eval_reporting_public_index.py`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH9_VALIDATION_LEDGER_20260330.md`
2. 实现 `scripts/ci/generate_eval_reporting_dashboard_payload.py`
3. 修改 workflow deploy-pages job
4. 补充/更新测试
5. 跑验证命令
6. 补齐该批 design/validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 10A 必跑命令

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_dashboard_payload.py \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 10A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH10_VALIDATION_LEDGER_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_DASHBOARD_PAYLOAD_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_DASHBOARD_PAYLOAD_SURFACE_ALIGNMENT_VALIDATION_20260331.md`

---

## Batch 10B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 10A ledger 和 validation MD
2. 实现 `scripts/ci/generate_eval_reporting_release_note_snippet.py`
3. 修改 workflow deploy-pages job
4. 新增/更新测试
5. 跑验证命令
6. 补齐该批 design/validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 10B 必跑命令

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_release_note_snippet.py \
  tests/unit/test_generate_eval_reporting_release_note_snippet.py

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_note_snippet.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 10B 合并回归

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py \
  tests/unit/test_generate_eval_reporting_release_note_snippet.py \
  tests/unit/test_generate_eval_reporting_public_index.py \
  tests/unit/test_generate_eval_reporting_release_summary.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_evaluation_report_workflow_release_summary.py -q
```

### Batch 10B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH10_VALIDATION_LEDGER_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_NOTE_SNIPPET_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_NOTE_SNIPPET_SURFACE_ALIGNMENT_VALIDATION_20260331.md`

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
   - `Batch 10A complete, stopped for validation`
   - 或 `Batch 10B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 10A accepted`

Claude 才能继续 Batch 10B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
