# Claude Collaboration Batch 17 Execution Runbook

日期：2026-04-01

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_DEVELOPMENT_PLAN_20260401.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_EXECUTION_RUNBOOK_20260401.md 执行。
现在只允许做 Batch 17A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md，
并明确停止，等待人工验证。
不允许提前做 Batch 17B。
```

---

## 总执行规则

1. 必须先做 Batch 17A
2. Batch 17A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 17B
4. 不允许越过 remove 直接做 merge
5. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 17A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
2. 盘点 `signature_policy` / `retry_plan` / `dry_run` 的当前代码与 workflow 引用
3. 删除 3 个 zero-consumer helper / consumer 与对应 tests
4. 更新 workflow step / sparse-checkout / artifact upload / summary append
5. 更新受影响的 workflow graph / pages deploy baseline tests
6. 编写 design / validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 17A 建议命令

```bash
rg -n "signature_policy|retry_plan|release_draft_dry_run|dry_run" \
  .github/workflows/evaluation-report.yml scripts/ci tests/unit

python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py

python3 -m pytest \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py -q
```

### Batch 17A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_VALIDATION_20260401.md`

---

## Batch 17B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 17A design / validation MD 和 ledger
2. 识别删除后需要固化的新 baseline
3. 更新 workflow graph / artifact naming / result-surface baseline tests
4. 编写 hardening design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 17B 建议命令

```bash
rg -n "signature_policy|retry_plan|release_draft_dry_run|dry_run" \
  .github/workflows/evaluation-report.yml tests/unit

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 17B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_VALIDATION_20260401.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 变更摘要
2. 修改文件 / 删除文件列表
3. 实际命令与结果
4. 未解决风险
5. 一句明确结论：
   - `Batch 17A complete, stopped for validation`
   - 或 `Batch 17B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 17A accepted`

Claude 才能继续 Batch 17B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
