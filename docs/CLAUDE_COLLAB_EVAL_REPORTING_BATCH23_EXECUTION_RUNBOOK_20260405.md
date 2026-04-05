# Claude Collaboration Batch 23 Execution Runbook

日期：2026-04-05

## 使用方式

把这份 Runbook 和对应的 Development Plan 一起交给 Claude。

推荐直接给 Claude 的指令：

```text
先阅读 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_DEVELOPMENT_PLAN_20260405.md，
再严格按 docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_EXECUTION_RUNBOOK_20260405.md 执行。
现在只允许做 Batch 23A。完成后必须更新
docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_VALIDATION_LEDGER_20260405.md，
并明确停止，等待人工验证。
不允许提前做 Batch 23B。
本轮不允许做任何代码改动。
```

---

## 总执行规则

1. 必须先做 Batch 23A
2. Batch 23A 完成后必须停止
3. 未收到继续指令前，不允许实现 Batch 23B
4. 不允许修改 workflow / scripts / tests
5. 不允许用 `workflow_dispatch` 冒充 full deploy-pages 验收
6. 每个批次结束后都必须更新 Validation Ledger

---

## Batch 23A 执行顺序

严格按以下顺序执行：

1. 阅读：
   - `.github/workflows/evaluation-report.yml`
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_DESIGN_20260403.md`
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_DESIGN_20260403.md`
2. 确认 `deploy-pages` 只在 `push/main` 条件下运行
3. 通过 GitHub CLI 找到 qualifying run：
   - workflow = `Evaluation Report`
   - branch = `main`
   - event = `push`
4. 记录 run-level evidence：
   - run id
   - url
   - head sha
   - conclusion
   - evaluate / deploy-pages jobs
5. 下载并检查 artifacts / summary / pages evidence
6. 编写 design / validation MD
7. 更新 Validation Ledger
8. 停止

### Batch 23A 建议命令

```bash
gh run list --workflow "Evaluation Report" --branch main --event push --limit 10

gh run view <RUN_ID>

gh run download <RUN_ID> --dir /tmp/eval-reporting-run-<RUN_ID>
```

可选辅助命令：

```bash
gh api repos/<OWNER>/<REPO>/actions/runs/<RUN_ID>/artifacts

gh api repos/<OWNER>/<REPO>/pages
```

### Batch 23A 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_VALIDATION_LEDGER_20260405.md`
- `docs/DEDUP_EVAL_REPORTING_E2E_GITHUB_ACTIONS_VERIFICATION_DESIGN_20260405.md`
- `docs/DEDUP_EVAL_REPORTING_E2E_GITHUB_ACTIONS_VERIFICATION_VALIDATION_20260405.md`

---

## Batch 23B 执行顺序

只有在人工明确允许后才能开始。

严格按以下顺序执行：

1. 阅读已完成的 Batch 23A design / validation MD 和 ledger
2. 归纳 closeout-ready / changes-required 结论
3. 归纳 residual risks（如有）
4. 编写 closeout design MD
5. 编写 closeout validation MD
6. 更新 Validation Ledger
7. 停止

### Batch 23B 建议命令

```bash
gh run view <RUN_ID>
```

### Batch 23B 完成后必须回填

- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_VALIDATION_LEDGER_20260405.md`
- `docs/DEDUP_EVAL_REPORTING_CLOSEOUT_DECISION_AND_RESIDUAL_RISK_DESIGN_20260405.md`
- `docs/DEDUP_EVAL_REPORTING_CLOSEOUT_DECISION_AND_RESIDUAL_RISK_VALIDATION_20260405.md`

---

## Claude 每批结束后的返回格式

Claude 返回内容必须包括：

1. 证据摘要
2. 实际命令与结果
3. 关键 run / artifact / pages 结论
4. blocker 或 residual risk
5. 一句明确结论：
   - `Batch 23A complete, stopped for validation`
   - 或 `Batch 23B complete, stopped for validation`

---

## 人工验证后的继续规则

只有当人工验证结果明确为：

- `Batch 23A accepted`

Claude 才能继续 Batch 23B。

如果人工指出问题，Claude 必须只修复该批文档或证据问题，不得越权进入下一批。
