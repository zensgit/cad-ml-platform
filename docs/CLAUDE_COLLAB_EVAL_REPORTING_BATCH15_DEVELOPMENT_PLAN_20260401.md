# Claude Collaboration Batch 15 Development Plan

日期：2026-04-01

## 目标概览

本轮协作开发只做 `eval reporting` 的 webhook resilience / signature policy 面，顺序固定：

1. 先生成一个给后续 retry queue / dead-letter surface 直接消费的 canonical retry plan artifact
2. 再生成一个给后续 signed delivery surface 直接消费的 canonical signature policy artifact

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_webhook_delivery_request.py`
  - `scripts/ci/post_eval_reporting_webhook_delivery.js`
  - `scripts/ci/generate_eval_reporting_webhook_export.py`
- 只能新增 thin retry-plan helper / thin signature-policy helper
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary / dashboard payload / webhook export / delivery request / delivery result
- retry plan 只能消费现有 webhook delivery result artifact
- signature policy 只能消费现有 webhook delivery request artifact

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/ci/eval_reporting_webhook_export.json`
  - `reports/ci/eval_reporting_webhook_delivery_request.json`
  - `reports/ci/eval_reporting_webhook_delivery_result.json`
  - `reports/ci/eval_reporting_release_draft_publish_result.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment / notify / Pages / status check / release-draft surfaces
  - external webhook delivery request surface
  - external webhook sender / delivery result surface
- 当前仍缺：
  - 一个专门给 future retry queue / dead-letter consumer 消费的 canonical retry plan artifact
  - 一个专门给 future signed webhook delivery consumer 消费的 canonical signature policy artifact

---

## Batch 15A：Webhook Retry / Dead-Letter Plan Surface

### 目标

生成一个给 retry queue、dead-letter surface、人工诊断直接消费的最小 canonical retry / dead-letter plan artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_webhook_retry_plan.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_webhook_retry_plan.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_webhook_delivery_result.json`
- 归一化为 retry / dead-letter 友好的最小 plan / policy

它不允许负责：

- 再次读取 webhook delivery request / webhook export / dashboard payload / release summary / public index / stack summary
- 重新发起 delivery
- 真正创建 retry queue / dead-letter queue
- 新增 queue worker / scheduler owner

#### retry / dead-letter 合同

建议默认输出：

- `reports/ci/eval_reporting_webhook_retry_plan.json`
- `reports/ci/eval_reporting_webhook_retry_plan.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_webhook_retry_plan"`
- `generated_at`
- `release_readiness`
- `delivery_mode`
- `delivery_attempted`
- `delivery_succeeded`
- `http_status`
- `delivery_error`
- `retry_recommended`
- `retry_policy`
- `retry_after_seconds`
- `retry_reason`
- `dead_letter_recommended`
- `dead_letter_reason`
- `retry_queue_target_kind`
- `retry_requires_explicit_enable`
- `source_delivery_result_surface_kind`

其中：

- `retry_recommended` 只能做对 delivery result 的 thin reuse / thin derivation
- `retry_policy` 只能是稳定 policy 值，例如：
  - `manual_or_future_queue`
  - `no_retry`
- `retry_after_seconds` 只能是 thin default recommendation，不能变成 scheduler owner
- `dead_letter_recommended` 只能是对 hard failure / no-retry 场景的薄推导
- `retry_queue_target_kind` 只能是 thin target descriptor，例如 `future_retry_queue`
- `retry_requires_explicit_enable` 必须固定表达“默认不入队”

Markdown 至少要包含：

- `Retry Recommended`
- `Retry Policy`
- `Retry After`
- `Dead Letter Recommended`
- `HTTP Status`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting webhook retry plan`
- always-run `Append eval reporting webhook retry plan to job summary`
- always-run `Upload eval reporting webhook retry plan`

这些步骤必须位于 webhook delivery result 之后。

### Batch 15A 验收条件

必须同时满足：

- retry plan 只消费 webhook delivery result artifact
- retry / dead-letter plan artifact 能稳定 materialize
- job summary 中出现 retry / dead-letter 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 15B：Webhook Signature Policy Surface

### 目标

生成一个给 future signed delivery、secret wiring、signature enforcement surface 直接消费的最小 canonical signature policy artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_webhook_signature_policy.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_webhook_signature_policy.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_webhook_delivery_request.json`
- 归一化为 signed webhook delivery 友好的最小 signature policy

它不允许负责：

- 再次读取 webhook delivery result / webhook export / dashboard payload / release summary / public index / stack summary
- 真正执行 HMAC / signing
- 读取真实 secret 或新增 secret manager owner
- 修改现有 sender / notify / PR comment / Pages owner

#### signature policy 合同

建议默认输出：

- `reports/ci/eval_reporting_webhook_signature_policy.json`
- `reports/ci/eval_reporting_webhook_signature_policy.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_webhook_signature_policy"`
- `generated_at`
- `webhook_event_type`
- `delivery_target_kind`
- `delivery_method`
- `signature_policy`
- `signature_required`
- `signature_algorithm`
- `signature_header_name`
- `signature_canonical_fields`
- `signing_enabled`
- `signature_requires_explicit_secret`
- `request_timeout_seconds`
- `source_delivery_request_surface_kind`

其中：

- `signature_policy` 只能是稳定 policy 值，例如：
  - `disabled_by_default`
  - `optional_future_hmac`
- `signature_required` 默认必须为 `false`
- `signing_enabled` 默认必须为 `false`
- `signature_algorithm` 只能是 thin declared algorithm，例如 `hmac-sha256`
- `signature_header_name` 只能是 thin header contract，例如 `X-Eval-Reporting-Signature`
- `signature_canonical_fields` 只能列出 future signer 应复用的字段，不得变成新的 content owner
- `signature_requires_explicit_secret` 必须固定表达“无 secret 时不能启用签名”

Markdown 至少要包含：

- `Signature Policy`
- `Signature Required`
- `Signature Algorithm`
- `Signature Header`
- `Delivery Method`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting webhook signature policy`
- always-run `Append eval reporting webhook signature policy to job summary`
- always-run `Upload eval reporting webhook signature policy`

这些步骤必须位于 webhook delivery request 之后。

#### 不做的事

本批不做：

- 真正的 retry queue / scheduler / dead-letter queue
- 真正的 webhook re-delivery loop
- 真正的 HMAC signing 或 secret lookup
- 修改现有 webhook sender 本体

### Batch 15B 验收条件

必须同时满足：

- signature policy 只消费 webhook delivery request artifact
- workflow 中存在独立的 signature policy 生成 / summary / upload steps
- default 路径不真正执行 signing
- signature policy artifact 可直接给后续 signed delivery surface 复用
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 15A

- 新增：
  - `tests/unit/test_generate_eval_reporting_webhook_retry_plan.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 15B

- 新增：
  - `tests/unit/test_generate_eval_reporting_webhook_signature_policy.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 15A

- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_RETRY_PLAN_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_RETRY_PLAN_ALIGNMENT_VALIDATION_20260401.md`

### Batch 15B

- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_SIGNATURE_POLICY_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_SIGNATURE_POLICY_ALIGNMENT_VALIDATION_20260401.md`

---

## 额外说明

- 本轮仍然遵守“先做 policy / plan surface，再做真正 queue / signing”的分层原则。
- 如果后续要做真正 retry queue、dead-letter queue、signature secret wiring 或 signed delivery，应作为下一批单独展开，而不是在 Batch 15 内顺手扩范围。
