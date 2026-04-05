# Claude Collaboration Batch 18 Execution Runbook

日期：2026-04-03

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_DEVELOPMENT_PLAN_20260403.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_EXECUTION_RUNBOOK_20260403.md 执行。
现在只允许做 Batch 18A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md，
并明确停止，等待人工验证。
不允许提前做 Batch 18B。
不允许触碰 release 链 merge。
```

---

## 总执行规则

1. 必须先做 Batch 18A
2. Batch 18A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 18B
4. 不允许越过 webhook merge 直接做 release merge
5. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 18A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
2. 盘点 `webhook_export` / `delivery_request` / `delivery_result` 的当前代码与 workflow 引用
3. 修改 `generate_eval_reporting_webhook_delivery_request.py`，改为直接读取 `dashboard_payload`
4. 删除 `webhook_export` helper 与对应 tests
5. 更新 workflow step / sparse-checkout / artifact upload / summary append
6. 更新受影响的 workflow graph / pages deploy / helper tests
7. 编写 design / validation MD
8. 更新 Validation Ledger
9. 停止

### Batch 18A 建议命令

```bash
rg -n "webhook_export|webhook_delivery_request|webhook_delivery_result" \
  .github/workflows/evaluation-report.yml scripts/ci tests/unit

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py

python3 -m pytest \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_post_eval_reporting_webhook_delivery_js.py -q
```

### Batch 18A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_VALIDATION_20260403.md`

---

## Batch 18B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 18A design / validation MD 和 ledger
2. 识别 webhook merge 后需要固化的新 baseline
3. 更新 workflow graph / artifact naming / result-surface baseline tests
4. 编写 hardening design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 18B 建议命令

```bash
rg -n "webhook_export|webhook_delivery_request|webhook_delivery_result" \
  .github/workflows/evaluation-report.yml tests/unit

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 18B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_VALIDATION_20260403.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 变更摘要
2. 修改文件 / 删除文件列表
3. 实际命令与结果
4. 未解决风险
5. 一句明确结论：
   - `Batch 18A complete, stopped for validation`
   - 或 `Batch 18B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 18A accepted`

Claude 才能继续 Batch 18B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
