# Claude Collaboration Batch 23 Development Plan

日期：2026-04-05

## 目标概览

本轮不再继续改代码或 workflow 结构，只做真实 GitHub Actions 运行态验收和最终 closeout。

执行顺序固定：

1. 先基于真实 GitHub-hosted run 收集 end-to-end 证据
2. 再给出最终 closeout 结论、残余风险和是否停止继续开 batch 的建议

执行原则：

- 本轮禁止修改 `.github/workflows/evaluation-report.yml`
- 本轮禁止修改 `scripts/ci/*`
- 本轮禁止修改 `tests/unit/*`
- 本轮只允许新增 / 更新 design / validation / ledger 文档
- 本轮禁止新增 artifact
- 本轮禁止以 `workflow_dispatch` 结果冒充 full deploy-pages 验收
- 本轮必须基于真实 GitHub-hosted run 给出证据

---

## 当前真实基线

截至 Batch 22 完成后，`eval reporting` 侧的 rationalization 与 consolidate 已全部闭环：

- webhook 链已收敛为：
  - `dashboard_payload -> delivery_request -> delivery_result`
- release 链已收敛为：
  - `dashboard_payload -> publish_result`
- `deploy-pages` 已收敛为：
  - all generates
  - consolidated summary
  - all uploads

当前最需要验证的不是代码结构，而是**真实 GitHub-hosted 运行是否与本地回归基线一致**。

关键前提必须写清：

- [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml) 的 `deploy-pages` job 条件是：
  - `github.ref == 'refs/heads/main' && github.event_name == 'push'`
- 因此：
  - `workflow_dispatch` 可以补充 evaluate-side 证据
  - 但**不能**作为 `deploy-pages / Pages / consolidated summary / upload block` 的 full E2E 替代

本轮 full closeout 的有效证据来源优先级：

1. 最新一个**包含 Batch 22 收口结果**的 `push` to `main` 的 `Evaluation Report` run
2. 如果不存在这样的 run：
   - 只能报告 blocker
   - 不得把 `workflow_dispatch` 误记为 full deploy-pages 验收

---

## Batch 23A：GitHub-Hosted E2E Verification

### 目标

针对真实 GitHub-hosted run 做 end-to-end 验收，确认收口后的 workflow 在 GitHub 上实际 materialize 出预期结果。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_E2E_GITHUB_ACTIONS_VERIFICATION_DESIGN_20260405.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_E2E_GITHUB_ACTIONS_VERIFICATION_VALIDATION_20260405.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_VALIDATION_LEDGER_20260405.md`

### 必做内容

必须完成：

1. 找到一个 qualifying run：
   - workflow = `Evaluation Report`
   - branch = `main`
   - event = `push`
   - run 对应的 workflow 内容已包含 Batch 22 consolidate
2. 记录并核对 run-level 证据：
   - run id / url / head sha / conclusion
   - evaluate job 结论
   - deploy-pages job 结论
3. 核对真实产物：
   - Pages URL / deployment URL
   - consolidated deploy-pages summary
   - artifact 列表
   - `eval_reporting_public_index`
   - `eval_reporting_dashboard_payload`
   - `eval_reporting_webhook_delivery_request`
   - `eval_reporting_webhook_delivery_result`
   - `eval_reporting_release_draft_publish_result`
4. 核对真实外部面：
   - status check
   - PR comment / notify 如当次 run 适用
5. 如果没有 qualifying push run：
   - 明确记录 blocker
   - 停止，不得伪造 full closeout

### 设计约束

- 不允许修改 workflow 或代码来“帮它通过”
- 不允许把 `workflow_dispatch` run 记作 full `deploy-pages` 验收
- 不允许只用本地 pytest 代替 GitHub-hosted 证据
- 可以下载 artifacts、查看 run logs、查看 Pages URL，但只能做证据收集

### Batch 23A 验收条件

必须同时满足：

- 有 qualifying GitHub-hosted `push/main` run，或明确 blocker
- 已记录 run id / url / head sha / conclusion
- 已核对 deploy-pages consolidated summary 与 artifact 列表
- 已核对 Pages / publish_result / delivery_result 等关键外部面
- design / validation / ledger 已回填

---

## Batch 23B：Closeout Decision And Residual Risk

### 目标

基于 Batch 23A 的真实运行证据，给出最终 closeout 结论：是否停止继续开 batch，是否仍有残余风险需要后续最小修复。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_CLOSEOUT_DECISION_AND_RESIDUAL_RISK_DESIGN_20260405.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_CLOSEOUT_DECISION_AND_RESIDUAL_RISK_VALIDATION_20260405.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH23_VALIDATION_LEDGER_20260405.md`

### 必做内容

至少应覆盖：

1. closeout 结论：
   - `closeout-ready`
   - 或 `changes-required`
2. residual risk 分类：
   - none
   - runtime-only issue
   - process/documentation issue
3. 如果存在问题：
   - 必须指向最小后续修复范围
   - 不得再次扩成大批次
4. 如果无问题：
   - 明确建议停止继续开 batch

### 设计约束

- 不允许借机重开新的 refactor / rationalization 主题
- closeout 结论必须绑定 Batch 23A 的真实运行证据
- 只能在 evidence 足够时给出 `closeout-ready`

### Batch 23B 验收条件

必须同时满足：

- closeout 结论与 23A 证据一致
- residual risk 明确、可执行
- 文档写清是否停止继续开 batch

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_DESIGN_20260403.md`

如有需要，可继续读取：

- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- GitHub-hosted run 的 artifacts / logs / pages / status evidence

---

## 不做的事

本轮明确不做：

- 不做代码改动
- 不做 workflow YAML 改动
- 不做测试改动
- 不做新 artifact 设计
- 不做新一轮 refactor

---

## 额外说明

- 本轮是整个 `eval reporting` rationalization / consolidate 主线的收口验收，不是新功能开发。
- 如果 Batch 23A/23B 结论为 `closeout-ready`，应明确建议停止继续开新 batch。
