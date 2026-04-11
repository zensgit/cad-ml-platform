# Claude Collaboration Batch 18 Development Plan

日期：2026-04-03

## 目标概览

本轮执行 Batch 16 目标架构中的 `Phase 2 webhook merge`，只压缩 webhook 链，不触碰 release 链。

执行顺序固定：

1. 先把 `webhook_export` 合并进 `delivery_request`
2. 再建立 merge 后的新 baseline，防止 `webhook_export` 重新出现

执行原则：

- 本轮允许修改 `.github/workflows/evaluation-report.yml`
- 本轮允许修改 `scripts/ci/generate_eval_reporting_webhook_delivery_request.py`
- 本轮允许删除 `webhook_export` 对应 helper / test / workflow step
- 本轮允许更新相关 workflow / pages deploy / helper tests
- 本轮必须补齐对应的 design / validation / ledger 文档
- 本轮禁止提前实现 release 链 merge
- 本轮禁止改变 `delivery_result` 的输入 / 输出 contract

---

## 当前真实基线

截至 Batch 17 完成后：

- `signature_policy`
- `retry_plan`
- `release_draft_dry_run`

已经删除。

当前 webhook 链最明显的中间层只剩：

- `eval_reporting_dashboard_payload`
- `eval_reporting_webhook_export`
- `eval_reporting_webhook_delivery_request`
- `eval_reporting_webhook_delivery_result`

其中 `webhook_export` 只是 `dashboard_payload` 的近似转抄层，Batch 16 已明确它应并入 `delivery_request`。

因此当前最低风险的下一步是：

- 让 `delivery_request` 直接读取 `eval_reporting_dashboard_payload.json`
- 删除 `webhook_export` 本身
- 保持 `delivery_result` 仍然只消费 `delivery_request`

本轮不做：

- `snippet/prefill/draft_payload/publish_payload -> publish_result` 合并
- `delivery_result` contract 调整
- 任何新的 public surface 变更

---

## Batch 18A：Webhook Export Merge Into Delivery Request

### 目标

落地 Batch 16 的 `Phase 2`：删除 `webhook_export` 这一中间层，让 `delivery_request` 直接消费 `dashboard_payload`。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_EXPORT_MERGE_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md`

### 必做代码范围

必须完成：

1. 修改 `generate_eval_reporting_webhook_delivery_request.py`
   - 改为直接读取 `eval_reporting_dashboard_payload.json`
2. 删除：
   - `generate_eval_reporting_webhook_export.py`
   - `tests/unit/test_generate_eval_reporting_webhook_export.py`
3. 更新 workflow：
   - 删除 `Generate/Append/Upload eval reporting webhook export`
   - 删除对应 sparse-checkout 项
   - 更新 `Generate eval reporting webhook delivery request` 的输入参数
4. 更新相关测试：
   - `test_generate_eval_reporting_webhook_delivery_request.py`
   - `test_evaluation_report_workflow_pages_deploy.py`
   - 任何直接断言 `webhook_export` 存在或顺序的测试

### 设计约束

- 不允许修改 `post_eval_reporting_webhook_delivery.js` 的输入 contract
- 不允许修改 `eval_reporting_webhook_delivery_result.json` 的 schema / field
- 不允许删除 `delivery_request` 或 `delivery_result`
- 不允许触碰 release 链 merge

### Batch 18A 验收条件

必须同时满足：

- `delivery_request` 已直接读取 `dashboard_payload`
- workflow 不再 materialize `webhook_export`
- sparse-checkout 不再包含 `generate_eval_reporting_webhook_export.py`
- `webhook_export` helper / tests 已删除
- 相关测试已更新并通过
- design / validation / ledger 已回填

---

## Batch 18B：Webhook Merge Baseline Hardening

### 目标

在 Batch 18A merge 完成后，建立新的 webhook baseline，防止 `webhook_export` 重新出现，同时保护 `delivery_request` / `delivery_result` 保持存在。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_DESIGN_20260403.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE2_WEBHOOK_BASELINE_HARDENING_VALIDATION_20260403.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH18_VALIDATION_LEDGER_20260403.md`

### 必做内容

至少应覆盖：

1. negative guard：
   - `webhook_export` 不得重新出现在 workflow step name / sparse-checkout / helper wiring
2. positive guard：
   - `delivery_request` 仍存在
   - `delivery_result` 仍存在
3. webhook merge 后的 workflow / artifact baseline
4. 回归测试命令和结果

### 设计约束

- 不允许新增新 artifact
- 不允许提前实现 release 链 merge
- 只能围绕 Batch 18A merge 后的 webhook baseline 做 hardening

### Batch 18B 验收条件

必须同时满足：

- merge 后的新 webhook baseline 有明确测试保护
- 文档写清 `webhook_export` 已被吸收
- 无越权 release merge / refactor

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_DESIGN_20260401.md`

如有需要，可继续读取：

- `scripts/ci/generate_eval_reporting_webhook_delivery_request.py`
- `scripts/ci/generate_eval_reporting_dashboard_payload.py`
- `scripts/ci/post_eval_reporting_webhook_delivery.js`
- `tests/unit/` 下与 workflow pages deploy / delivery request / delivery result 相关的测试

---

## 不做的事

本轮明确不做：

- 不做 release 链 merge
- 不做 `publish_result` contract 调整
- 不新增任何 artifact
- 不改变 public index / dashboard payload / delivery result 的对外 contract
- 不做大范围 workflow 重排

---

## 额外说明

- 本轮是 `Phase 1 removes` 之后的第一轮 merge 落地。
- 如果 Batch 18A/18B 结论明确，下一轮才进入 release 链 merge。
