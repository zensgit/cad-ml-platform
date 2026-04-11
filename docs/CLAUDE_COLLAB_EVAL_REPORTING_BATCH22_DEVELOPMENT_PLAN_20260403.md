# Claude Collaboration Batch 22 Development Plan

日期：2026-04-03

## 目标概览

本轮开始执行 `deploy-pages` workflow consolidate。数据流已经稳定，下一步不再继续改 surface，而是收口重复的 summary / ordering 样板。

执行顺序固定：

1. 先把 `deploy-pages` 中 5 个 per-surface summary append step 收口成 1 个 consolidated summary step
2. 再建立 consolidate 后的新 workflow baseline，防止旧的 per-surface summary step 重新出现

执行原则：

- 本轮允许修改 `.github/workflows/evaluation-report.yml`
- 本轮允许修改 `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- 本轮必须补齐对应的 design / validation / ledger 文档
- 本轮禁止新增 artifact
- 本轮禁止改变任何 artifact schema / field
- 本轮禁止改变 upload artifact naming / path contract
- 本轮禁止触碰 release / webhook dataflow

---

## 当前真实基线

截至 Batch 21 完成后，`deploy-pages` job 的 post-deploy surface 已稳定为：

- `eval_reporting_public_index`
- `eval_reporting_dashboard_payload`
- `eval_reporting_webhook_delivery_request`
- `eval_reporting_webhook_delivery_result`
- `eval_reporting_release_draft_publish_result`

当前 job 结构包含：

- 5 个 `Generate ...`
- 5 个 `Append ... to job summary`
- 5 个 `Upload ...`

其中 5 个 summary append step 只是重复的 shell 样板：

- 定义 `*_MD`
- 写 section title
- `if [[ -f ]]` 则 `cat`
- 否则写 fallback line

这些步骤不改变数据流，只增加 YAML 体积和维护成本。因此当前最低风险的下一步是：

- 保留所有 generate / upload steps
- 把 5 个 per-surface summary append step 合并为 1 个 consolidated summary step
- 固化新的 ordering：
  - all generate steps
  - consolidated summary
  - all upload steps

本轮不做：

- artifact merge
- upload artifact name 改动
- helper / consumer 输入输出改动
- webhook / release 任何新的 merge
- evaluate job 改动

---

## Batch 22A：Deploy-Pages Workflow Consolidation

### 目标

收口 `deploy-pages` 中重复的 summary append 样板，只保留 1 个 consolidated summary step，同时保持现有 artifact / upload contract 不变。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_DEPLOY_PAGES_WORKFLOW_CONSOLIDATION_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`

### 必做代码范围

必须完成：

1. 修改 `.github/workflows/evaluation-report.yml`
   - 删除 5 个 per-surface summary append step：
     - `Append public URLs to job summary`
     - `Append eval reporting dashboard payload to job summary`
     - `Append eval reporting webhook delivery request to job summary`
     - `Append eval reporting webhook delivery result to job summary`
     - `Append eval reporting release draft publish result to job summary`
   - 新增 1 个 consolidated summary step
   - consolidated step 必须按固定顺序写入：
     - public index
     - dashboard payload
     - delivery request
     - delivery result
     - publish result
2. 保持以下不变：
   - 5 个 generate step
   - 5 个 upload step
   - 所有 artifact 输出路径
   - 所有 artifact upload 名称
3. 更新相关测试：
   - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
   - 任何直接断言旧 per-surface summary step 名称存在的测试

### 设计约束

- 不允许创建新的 helper / script 只是为了拼 summary
- consolidated summary 只能在 workflow shell 中完成
- 不允许改变 `reports/ci/*.json` / `reports/ci/*.md` 的生成逻辑
- 不允许改变 upload artifact contract
- 不允许改动 `deploy-pages` 前 4 个 Pages infra step

### Batch 22A 验收条件

必须同时满足：

- `deploy-pages` 中只剩 1 个 consolidated summary step
- 旧 5 个 per-surface summary step 已删除
- 5 个 upload step 仍全部存在
- consolidated summary 的 section 顺序固定
- 相关测试已更新并通过
- design / validation / ledger 已回填

---

## Batch 22B：Consolidated Workflow Baseline Hardening

### 目标

在 Batch 22A consolidate 完成后，建立新的 workflow baseline，防止旧的 per-surface summary step 回流，同时保护 consolidated summary 和 upload block 顺序。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE6_CONSOLIDATED_DEPLOY_PAGES_BASELINE_HARDENING_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH22_VALIDATION_LEDGER_20260403.md`

### 必做内容

至少应覆盖：

1. negative guard：
   - 旧 5 个 per-surface summary step 名称不得重新出现
2. positive guard：
   - consolidated summary step 仍存在
   - 5 个 upload step 仍存在
3. ordering guard：
   - consolidated summary 在最后一个 generate step 之后
   - consolidated summary 在第一个 upload step 之前
   - upload block 顺序固定
4. consolidate 后的 workflow baseline
5. 回归测试命令和结果

### 设计约束

- 不允许新增新 artifact
- 不允许再做 upload artifact regroup / rename
- 只能围绕 Batch 22A consolidate 后的 workflow baseline 做 hardening

### Batch 22B 验收条件

必须同时满足：

- consolidate 后的新 workflow baseline 有明确测试保护
- 文档写清旧 summary step 已被 consolidated summary 吸收
- 无越权 artifact / dataflow 改动

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH21_VALIDATION_LEDGER_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_RELEASE_DRAFT_PAYLOAD_MERGE_DESIGN_20260403.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE5_FINAL_RELEASE_BASELINE_HARDENING_DESIGN_20260403.md`

如有需要，可继续读取：

- `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 不做的事

本轮明确不做：

- 不做 artifact merge / rename
- 不做 webhook / release consumer 逻辑改动
- 不做 evaluate job consolidate
- 不新增任何 artifact
- 不改变任何 surface contract

---

## 额外说明

- 本轮只收 YAML，不再收数据流。
- 如果 Batch 22A/22B 结论明确，下一轮才评估是否需要进一步压缩 upload block，或到此收口。
