# Claude Collaboration Batch 17 Development Plan

日期：2026-04-01

## 目标概览

本轮开始执行 Batch 16 已确认的 rationalization 落地，先做最低风险的 `Phase 1 removes`，不做任何 merge。

执行顺序固定：

1. 先删除 3 个 `zero runtime consumer` surface
2. 再做删除后的 workflow / test baseline hardening

执行原则：

- 本轮允许修改 `.github/workflows/evaluation-report.yml`
- 本轮允许删除不再需要的 `scripts/ci/` helper / consumer
- 本轮允许删除或更新相关测试
- 本轮必须补齐对应的 design / validation / ledger 文档
- 本轮禁止提前实现 Batch 16B 中定义的 merge 项
- 本轮禁止改变保留 surface 的 contract

---

## 当前真实基线

Batch 16 已明确：

- `signature_policy`
- `retry_plan`
- `release_draft_dry_run`

这 3 个 artifact 当前都没有真实 runtime consumer，属于 `zero consumer / future intent` surface。

因此最小、最低风险的第一步是：

- 删除这 3 个 surface 本身
- 删除它们在 workflow 中对应的 generate / append / upload step
- 删除它们的 sparse-checkout 项
- 删除不再需要的 helper / consumer 文件与定向测试

本轮不做：

- `webhook_export -> delivery_request` 合并
- `snippet/prefill/draft_payload/publish_payload -> publish_result` 合并
- 任何 public surface contract 调整

---

## Batch 17A：Phase 1 Zero-Consumer Surface Removal

### 目标

落地 Batch 16 的 Phase 1：删除 3 个零 consumer surface，并把 workflow / tests / docs 同步到新的最小基线。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_DESIGN_20260401.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE1_ZERO_CONSUMER_SURFACE_REMOVAL_VALIDATION_20260401.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`

### 必做代码范围

必须删除或停止 materialize 以下 3 个 surface：

1. `eval_reporting_webhook_signature_policy`
2. `eval_reporting_webhook_retry_plan`
3. `eval_reporting_release_draft_dry_run`

必须同步处理：

- 对应 helper / consumer 文件
- 对应 unit tests
- `.github/workflows/evaluation-report.yml` 中的 step / artifact upload / summary append / sparse-checkout
- 受影响的 workflow graph / pages deploy / inventory baseline tests

### 设计约束

- 不允许删除 `eval_reporting_webhook_delivery_result`
- 不允许删除 `eval_reporting_release_draft_publish_result`
- 不允许把 `delivery_request` 改成直接消费 `dashboard_payload`
- 不允许把 `publish_result` 改成直接消费 `dashboard_payload`
- 本批只做 remove，不做 merge

### Batch 17A 验收条件

必须同时满足：

- 3 个 zero-consumer helper / consumer 已从代码和 workflow 中移除
- workflow 不再引用这 3 个 artifact 名称
- sparse-checkout 不再包含这 3 个脚本
- 删除后的相关测试已更新并通过
- design / validation / ledger 已回填

---

## Batch 17B：Phase 1 Baseline Hardening

### 目标

在 Batch 17A 删除完成后，建立新的 workflow / artifact baseline，避免后续 batch 继续回归到已删除 surface。

### 必做产出

1. 新增 design MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_DESIGN_20260401.md`
2. 新增 validation MD：
   - `docs/DEDUP_EVAL_REPORTING_PHASE1_BASELINE_HARDENING_VALIDATION_20260401.md`
3. 更新：
   - `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH17_VALIDATION_LEDGER_20260401.md`

### 必做内容

至少应覆盖：

1. workflow graph / pages deploy tests 的新基线
2. artifact inventory / naming baseline 的新基线
3. 删除后仍保留的 release / webhook result surface 的 contract 防回归
4. 回归测试命令和结果

### 设计约束

- 不允许新增新 artifact
- 不允许提前实现 merge 方案
- 只能围绕 Batch 17A 删除后的基线做 hardening

### Batch 17B 验收条件

必须同时满足：

- 删除后的 workflow / artifact 基线有明确测试保护
- 文档写清新的最小保留面
- 无越权 merge / refactor

---

## 必须阅读的输入

Claude 在执行本批前必须阅读：

- `.github/workflows/evaluation-report.yml`
- `docs/CLAUDE_COLLAB_EVAL_REPORTING_BATCH16_VALIDATION_LEDGER_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WORKFLOW_RATIONALIZATION_TARGET_ARCHITECTURE_VALIDATION_20260401.md`

如有需要，可继续读取：

- `scripts/ci/` 下与 `signature_policy` / `retry_plan` / `dry_run` 相关的 helper / consumer
- `tests/unit/` 下相关 workflow / pages deploy / JS wrapper tests

---

## 不做的事

本轮明确不做：

- 不做 `webhook_export` 合并
- 不做 `release_note_snippet` / `prefill` / `draft_payload` / `publish_payload` 合并
- 不新增任何 artifact
- 不修改 public index / dashboard payload / delivery result / publish result 的对外 contract
- 不做大范围 workflow 重排

---

## 额外说明

- 本轮是从 design-only rationalization 转入代码落地的第一轮。
- 如果 Batch 17A/17B 结论明确，下一轮才进入 `Phase 2 webhook merge`。
