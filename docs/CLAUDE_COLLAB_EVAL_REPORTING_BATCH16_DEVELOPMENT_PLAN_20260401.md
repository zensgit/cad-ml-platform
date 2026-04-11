# Claude Collaboration Batch 16 Development Plan

日期：2026-04-01

## 目标概览

本轮协作开发不再新增任何 `eval reporting` artifact，也不修改 workflow / helper / test 代码。

本轮只做 design-only 的收缩评审，顺序固定：

1. 先完成完整的 artifact inventory / consumer map / owner-boundary review
2. 再产出 keep / merge / remove / move-out-of-deploy-pages 决策和简化后的目标 workflow 结构

执行原则：

- 本轮禁止新增 Python / JS helper
- 本轮禁止修改 `.github/workflows/evaluation-report.yml`
- 本轮禁止修改 `scripts/ci/` 下任何业务逻辑
- 本轮禁止修改测试文件
- 本轮只允许新增或更新 `docs/` 下的 design / validation / ledger 文档
- 所有结论必须基于当前仓库真实代码与 workflow，而不是抽象想象

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已拥有一条很长的 `deploy-pages` artifact 生成链
- 当前主要 artifact 包括但不限于：
  - `eval_reporting_bundle`
  - `eval_reporting_bundle_health_report`
  - `eval_reporting_index`
  - `eval_reporting_stack_summary`
  - `eval_reporting_release_summary`
  - `eval_reporting_public_index`
  - `eval_reporting_dashboard_payload`
  - `eval_reporting_release_note_snippet`
  - `eval_reporting_release_draft_prefill`
  - `eval_reporting_webhook_export`
  - `eval_reporting_webhook_delivery_request`
  - `eval_reporting_webhook_signature_policy`
  - `eval_reporting_webhook_delivery_result`
  - `eval_reporting_webhook_retry_plan`
  - `eval_reporting_release_draft_payload`
  - `eval_reporting_release_draft_dry_run`
  - `eval_reporting_release_draft_publish_payload`
  - `eval_reporting_release_draft_publish_result`
- 当前的真实风险不是“能力缺失”，而是：
  - workflow step 数量持续增长
  - 多个 artifact 只是上游 JSON 的 thin pass-through / thin re-label
  - 部分 surface 的独立 consumer 价值开始弱化

---

## Batch 16A：Artifact Inventory / Consumer Map / Boundary Review

### 目标

对当前 `eval reporting` stack 做一次仓库落地事实盘点，明确每个 artifact 的：

- owner / source
- downstream consumer
- 是否属于 public surface / action result / thin pass-through
- 是否继续值得保留在主 workflow 中

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_ARTIFACT_INVENTORY_AND_CONSUMER_MAP_DESIGN_20260401.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_ARTIFACT_INVENTORY_AND_CONSUMER_MAP_VALIDATION_20260401.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`

### 必做内容

Artifact inventory 文档必须至少覆盖这些列：

- `artifact`
- `produced_by`
- `primary_input`
- `current_consumer`
- `classification`
- `recommended_action`

其中：

- `classification` 只能用受控分类，例如：
  - `owner`
  - `public_surface`
  - `action_result`
  - `delivery_surface`
  - `thin_pass_through`
- `recommended_action` 只能用受控决策，例如：
  - `keep`
  - `merge`
  - `remove`
  - `move_out_of_deploy_pages`
  - `defer_decision`

### 设计约束

- 必须基于真实 workflow 和当前脚本依赖链，不允许“凭感觉”列 consumer
- 必须特别审查以下链路是否存在高重复：
  - `dashboard -> release note snippet -> release draft prefill -> release draft payload -> publish payload`
  - `webhook export -> delivery request -> signature policy`
  - `delivery result -> retry plan`
- 必须明确标出哪些 artifact 具有真实独立 consumer，哪些只是未来意图 surface
- 本批不做任何 keep/merge/remove 的最终落地代码实现，只做 inventory 和 evidence

### Batch 16A 验收条件

必须同时满足：

- inventory 覆盖当前主 workflow 中所有 eval reporting artifact
- 每个 artifact 都有明确 consumer 或明确写出“当前无独立 consumer”
- 文档中明确区分 owner / public surface / action result / thin pass-through
- validation MD 给出盘点方法和证据来源
- 无代码改动

---

## Batch 16B：Workflow Rationalization / Target Architecture

### 目标

在 Batch 16A inventory 的基础上，给出下一阶段应执行的收缩方案，而不是继续加新 surface。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_VALIDATION_20260401.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`

### 必做内容

Batch 16B 文档必须至少包含：

1. `keep list`
2. `merge candidates`
3. `remove candidates`
4. `move-out-of-deploy-pages candidates`
5. `target workflow shape`
6. `migration order`
7. `risk / rollback notes`

### 设计约束

- 必须明确指出：
  - 哪些 artifact 应继续保留为长期稳定 surface
  - 哪些 artifact 应并入上游 owner 或上游 result
  - 哪些 artifact 不应继续留在 `deploy-pages`
- 必须优先保护：
  - root / public discovery surface
  - health / stack summary surface
  - 真正有动作语义的 result surface
- 必须优先审查并可能压缩：
  - release 链上的中间层
  - webhook 链上的中间层
  - 纯 policy / future-intent artifact
- 本批仍然不做任何代码改动，只输出目标架构和 cut plan

### Batch 16B 验收条件

必须同时满足：

- 给出清晰的 keep / merge / remove / move 清单
- 给出简化后的目标 workflow 结构
- 给出可执行的迁移顺序
- 明确写出“本批无代码改动”

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH14_VALIDATION_LEDGER_20260401.md`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH15_VALIDATION_LEDGER_20260401.md`

如有需要，可继续读取：

- `scripts/ci/` 下各 artifact helper / consumer 的当前实现

---

## 不做的事

本轮明确不做：

- 不新增任何 artifact
- 不修改 workflow step 顺序
- 不删除任何现有 step
- 不修改 helper / consumer 逻辑
- 不补测试
- 不做自动 refactor

---

## 额外说明

- 本轮是“先收缩设计，再收缩实现”的转折点。
- 如果 Batch 16 结论明确，下一轮应是 workflow / artifact rationalization 的代码落地，而不是继续堆新的 pass-through surface。
