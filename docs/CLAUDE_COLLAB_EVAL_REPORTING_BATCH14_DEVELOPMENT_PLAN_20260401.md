# Claude Collaboration Batch 14 Development Plan

日期：2026-04-01

## 目标概览

本轮协作开发只做 `eval reporting` 的 external webhook delivery 面，顺序固定：

1. 先生成一个给 webhook sender 直接消费的 canonical delivery request / policy artifact
2. 再基于这份 request 实现默认关闭、显式 gate 才尝试执行的 external webhook sender + fail-soft delivery result / retry-friendly surface

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_webhook_export.py`
  - `scripts/ci/generate_eval_reporting_dashboard_payload.py`
  - `scripts/ci/generate_eval_reporting_public_index.py`
- 只能新增 thin delivery-request helper / thin delivery consumer
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary / dashboard payload / webhook export / release-note snippet / release-draft prefill / release-draft payload
- delivery request 只能消费现有 webhook export artifact
- webhook sender 只能消费新的 delivery request artifact

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/ci/eval_reporting_dashboard_payload.json`
  - `reports/ci/eval_reporting_webhook_export.json`
  - `reports/ci/eval_reporting_release_draft_prefill.json`
  - `reports/ci/eval_reporting_release_draft_payload.json`
  - `reports/ci/eval_reporting_release_draft_publish_result.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
  - `Eval Reporting` GitHub status check
  - release-draft prefill / payload / dry-run / publish automation surface
- 当前仍缺：
  - 一个专门给 external webhook sender 消费的 canonical delivery request / policy artifact
  - 一个真正的 optional external webhook sender / delivery result surface
  - 一个给后续 retry surface 直接复用的稳定 delivery result contract

---

## Batch 14A：Webhook Delivery Request / Policy Surface

### 目标

生成一个给 external webhook sender、后续 delivery gate、人工诊断直接消费的最小 canonical delivery request / policy artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_webhook_delivery_request.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_webhook_delivery_request.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_webhook_export.json`
- 归一化为 external webhook sender 友好的最小 delivery request / policy

它不允许负责：

- 再次读取 dashboard payload / release summary / public index / stack summary
- 重新生成 webhook export
- 真正发出 HTTP 请求
- 新增 queue / retry loop / dead-letter owner

#### request / policy 合同

建议默认输出：

- `reports/ci/eval_reporting_webhook_delivery_request.json`
- `reports/ci/eval_reporting_webhook_delivery_request.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_webhook_delivery_request"`
- `generated_at`
- `webhook_event_type`
- `ingestion_schema_version`
- `release_readiness`
- `stack_status`
- `dashboard_headline`
- `missing_count`
- `stale_count`
- `mismatch_count`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `delivery_target_kind`
- `delivery_method`
- `delivery_policy`
- `delivery_allowed`
- `delivery_requires_explicit_enable`
- `request_timeout_seconds`
- `request_body_json`
- `source_webhook_export_surface_kind`

其中：

- `delivery_target_kind` 只能是 thin target descriptor，例如 `external_webhook`
- `delivery_method` 必须固定为稳定值，例如 `POST`
- `delivery_policy` 只能是 thin policy，例如：
  - `disabled_by_default`
  - `artifact_available_only`
- `delivery_allowed` 只能由现有 webhook export 是否可用做薄推导，不能新建复杂 gate
- `delivery_requires_explicit_enable` 必须固定表达“默认不发送”
- `request_body_json` 只能是 webhook export 的稳定复用体，不得变成新的 owner schema

Markdown 至少要包含：

- `Webhook Event`
- `Delivery Policy`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting webhook delivery request`
- always-run `Append eval reporting webhook delivery request to job summary`
- always-run `Upload eval reporting webhook delivery request`

这些步骤必须位于 webhook export 之后，且在真正 sender step 之前。

### Batch 14A 验收条件

必须同时满足：

- delivery request 只消费 webhook export artifact
- request / policy artifact 能稳定 materialize
- job summary 中出现 delivery policy 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 14B：Optional External Webhook Sender / Delivery Result Surface

### 目标

让 `evaluation-report.yml` 基于 delivery request 实现一个默认关闭、显式 gate 才尝试执行的 external webhook sender，并产出 fail-soft delivery result / retry-friendly artifact。

### 必做改动

1. 新增 `scripts/ci/post_eval_reporting_webhook_delivery.js` 或等价 thin consumer
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### consumer 职责

`post_eval_reporting_webhook_delivery.js` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_webhook_delivery_request.json`
- 在显式 enable 且 gate 允许时，可选向外部 webhook 发起一次 delivery
- 生成稳定的 delivery result artifact

它不允许负责：

- 再次读取 webhook export / dashboard payload / release summary / public index / stack summary
- 重新生成 delivery request
- 新增复杂 retry loop / queue / dead-letter owner
- 修改 notify / PR comment / Pages / release draft owner

#### delivery result 合同

建议默认输出：

- `reports/ci/eval_reporting_webhook_delivery_result.json`
- `reports/ci/eval_reporting_webhook_delivery_result.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_webhook_delivery_result"`
- `generated_at`
- `release_readiness`
- `delivery_enabled`
- `delivery_allowed`
- `delivery_attempted`
- `delivery_succeeded`
- `delivery_mode`
- `delivery_target_kind`
- `webhook_event_type`
- `http_status`
- `delivery_error`
- `retry_recommended`
- `retry_hint`
- `request_timeout_seconds`

其中：

- 默认必须 `delivery_enabled = false`
- 只有在显式 gate 满足时才允许尝试 delivery
- 无 URL / 无权限 / HTTP 失败 / timeout 场景必须 fail-soft，不得破坏主评估链
- 若 delivery 未尝试或失败，artifact 仍必须稳定落盘
- 本批的 “retry surface” 只允许体现为稳定的 result fields，不允许直接实现 retry loop

Markdown 至少要包含：

- `Delivery Attempted`
- `Delivery Succeeded`
- `Delivery Mode`
- `HTTP Status`
- `Retry Recommended`

#### workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting webhook delivery result`
- always-run `Append eval reporting webhook delivery result to job summary`
- always-run `Upload eval reporting webhook delivery result`

这些步骤必须位于 delivery request 之后，且默认不做实际发送。

#### 不做的事

本批不做：

- 自动 retry queue / dead-letter queue
- webhook signature / HMAC policy owner
- 自动创建 GitHub release
- 修改 notify / PR comment / Pages owner

### Batch 14B 验收条件

必须同时满足：

- webhook sender 只消费 delivery request artifact
- workflow 中存在独立的 delivery result / optional sender steps
- default 路径不真正发送外部 webhook
- delivery result 可直接给后续 retry surface 复用
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 14A

- 新增：
  - `tests/unit/test_generate_eval_reporting_webhook_delivery_request.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 14B

- 新增：
  - `tests/unit/test_post_eval_reporting_webhook_delivery_js.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 14A

- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_DELIVERY_REQUEST_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_DELIVERY_REQUEST_ALIGNMENT_VALIDATION_20260401.md`

### Batch 14B

- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_DELIVERY_AUTOMATION_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_DELIVERY_AUTOMATION_ALIGNMENT_VALIDATION_20260401.md`

---

## 额外说明

- 本轮仍然遵守“先做 surface，再做真正 delivery”的分层原则。
- 如果后续要做真正 retry queue / dead-letter / signature policy，应作为下一批单独展开，而不是在 Batch 14 内顺手扩范围。
