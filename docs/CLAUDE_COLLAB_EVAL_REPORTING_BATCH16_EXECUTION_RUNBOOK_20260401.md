# Claude Collaboration Batch 16 Execution Runbook

日期：2026-04-01

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_DEVELOPMENT_PLAN_20260401.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_EXECUTION_RUNBOOK_20260401.md 执行。
现在只允许做 Batch 16A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md，
并明确停止，等待人工验证。
不允许提前做 Batch 16B。
本轮不允许做任何代码改动。
```

---

## 总执行规则

1. 必须先做 Batch 16A
2. Batch 16A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 16B
4. 不允许修改 Development Plan 中写死的接口和范围
5. 本轮禁止修改任何 `.py` / `.js` / `.yml` / `tests/` 文件
6. 本轮只允许新增或更新 `docs/` 下的文档
7. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 16A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH14_VALIDATION_LEDGER_20260401.md`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH15_VALIDATION_LEDGER_20260401.md`
2. 盘点当前所有 `eval reporting` artifact
3. 梳理每个 artifact 的真实 consumer / boundary
4. 编写 inventory / consumer map design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 16A 建议命令

```bash
rg -n "eval_reporting_[A-Za-z0-9_]+\\.(json|md)|name: (Generate|Append|Upload|Post) eval reporting" \
  .github/workflows/evaluation-report.yml

rg -n "eval_reporting_(stack_summary|release_summary|public_index|dashboard_payload|release_note_snippet|release_draft_prefill|webhook_export|webhook_delivery_request|webhook_signature_policy|webhook_delivery_result|webhook_retry_plan|release_draft_payload|release_draft_dry_run|release_draft_publish_payload|release_draft_publish_result)" \
  scripts tests .github/workflows/evaluation-report.yml
```

### Batch 16A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_ARTIFACT_INVENTORY_AND_CONSUMER_MAP_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_ARTIFACT_INVENTORY_AND_CONSUMER_MAP_VALIDATION_20260401.md`

---

## Batch 16B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 16A design / validation MD 和 ledger
2. 归纳 keep / merge / remove / move-out-of-deploy-pages 决策
3. 绘制简化后的目标 workflow 结构
4. 编写 target architecture design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 16B 建议命令

```bash
rg -n "eval_reporting_[A-Za-z0-9_]+\\.(json|md)|name: (Generate|Append|Upload|Post) eval reporting" \
  .github/workflows/evaluation-report.yml

sed -n '1750,3180p' .github/workflows/evaluation-report.yml
```

### Batch 16B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_VALIDATION_20260401.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 本批结论摘要
2. 新增/更新文档列表
3. 实际命令
4. 实际结果
5. 关键发现
6. keep / merge / remove / move 建议
7. 一句明确结论：
   - `Batch 16A complete, stopped for validation`
   - 或 `Batch 16B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 16A accepted`

Claude 才能继续 Batch 16B。

如果人工指出问题，Claude 必须只修复该批文档问题，不得越权进入下一批。
