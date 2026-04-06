# Claude Collaboration Batch 21 Execution Runbook

日期：2026-04-03

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_DEVELOPMENT_PLAN_20260403.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_EXECUTION_RUNBOOK_20260403.md 执行。
现在只允许做 Batch 21A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md，
并明确停止，等待人工验证。
不允许提前做 Batch 21B。
不允许触碰 workflow consolidate。
```

---

## 总执行规则

1. 必须先做 Batch 21A
2. Batch 21A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 21B
4. 不允许越过 final release merge 直接做 workflow consolidate
5. 不允许修改 Development Plan 中写死的接口和范围
6. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 21A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH20_VALIDATION_LEDGER_20260403.md`
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
2. 盘点 `dashboard_payload` / `draft_payload` / `publish_result` 的当前代码与 workflow 引用
3. 修改 `post_eval_reporting_release_draft_publish.js`，改为直接读取 `dashboard_payload`
4. 删除 `draft_payload` helper 与对应 tests
5. 更新 workflow step / sparse-checkout / artifact upload / summary append
6. 更新受影响的 workflow pages deploy / publish-result tests
7. 编写 design / validation MD
8. 更新 Validation Ledger
9. 停止

### Batch 21A 建议命令

```bash
rg -n "dashboard_payload|release_draft_payload|release_draft_publish_result" \
  .github/workflows/evaluation-report.yml scripts/ci tests/unit

node --check scripts/ci/post_eval_reporting_release_draft_publish.js

python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py -q

rg -n "release_draft_payload|draftPayloadPath" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/post_eval_reporting_release_draft_publish.js
```

### Batch 21A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_VALIDATION_20260403.md`

---

## Batch 21B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 21A design / validation MD 和 ledger
2. 识别 final release merge 后需要固化的新 baseline
3. 更新 workflow graph / artifact naming / result-surface baseline tests
4. 编写 hardening design MD
5. 编写 validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 21B 建议命令

```bash
rg -n "dashboard_payload|release_draft_payload|release_draft_publish_result" \
  .github/workflows/evaluation-report.yml tests/unit

python3 -m pytest tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
```

### Batch 21B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_VALIDATION_20260403.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 变更摘要
2. 修改文件 / 删除文件列表
3. 实际命令与结果
4. 未解决风险
5. 一句明确结论：
   - `Batch 21A complete, stopped for validation`
   - 或 `Batch 21B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 21A accepted`

Claude 才能继续 Batch 21B。

如果人工指出问题，Claude 必须只修复该批问题，不得越权进入下一批。
