# Claude Collaboration Batch 22 Execution Runbook

日期：2026-04-03

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_DEVELOPMENT_PLAN_20260403.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_EXECUTION_RUNBOOK_20260403.md 执行。
现在只允许做 Batch 22A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md，
并明确停止，等待人工验证。
不允许提前做 Batch 22B。
不允许触碰 artifact contract。
```

---

## 总执行规则

1. 必须先做 Batch 22A
2. Batch 22A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 22B
4. 不允许越过 workflow consolidate 去改 artifact / dataflow
5. 不允许修改 Development Plan 中写死的接口和范围
6. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 22A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
2. 盘点 `deploy-pages` 中现有 5 个 per-surface summary step
3. 修改 workflow，把 5 个 summary append step 合并成 1 个 consolidated summary step
4. 更新 `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
5. 编写 design / validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 22A 建议命令

```bash
rg -n "Append public URLs to job summary|Append eval reporting dashboard payload to job summary|Append eval reporting webhook delivery request to job summary|Append eval reporting webhook delivery result to job summary|Append eval reporting release draft publish result to job summary" \
  .github/workflows/evaluation-report.yml tests/unit

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 22A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_VALIDATION_20260403.md`

---

## Batch 22B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 22A design / validation MD 和 ledger
2. 识别 consolidate 后需要固化的新 workflow baseline
3. 更新 workflow graph / ordering / upload baseline tests
4. 编写 hardening design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 22B 建议命令

```bash
rg -n "Consolidated eval reporting deploy-pages summary|Append public URLs to job summary|Append eval reporting dashboard payload to job summary|Append eval reporting webhook delivery request to job summary|Append eval reporting webhook delivery result to job summary|Append eval reporting release draft publish result to job summary" \
  .github/workflows/evaluation-report.yml tests/unit

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 22B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_VALIDATION_20260403.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 变更摘要
2. 修改文件列表
3. 实际命令与结果
4. 未解决风险
5. 一句明确结论：
   - `Batch 22A complete, stopped for validation`
   - 或 `Batch 22B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 22A accepted`

Claude 才能继续 Batch 22B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
