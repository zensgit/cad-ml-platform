# Claude Collaboration Batch 11 Development Plan

日期：2026-03-31

## 目标概览

本轮协作开发只做 `eval reporting` 的更外层交付消费面，顺序固定：

1. 先生成一个给 release manager / handoff owner 直接使用的 release-draft prefill artifact
2. 再基于稳定的 dashboard payload 生成 external dashboard webhook / ingestion export surface

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_dashboard_payload.py`
  - `scripts/ci/generate_eval_reporting_release_note_snippet.py`
  - `scripts/ci/generate_eval_reporting_public_index.py`
- 只能新增 thin release-draft helper / thin webhook-export helper
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary / dashboard payload
- release-draft surface 只能消费现有 release-note snippet artifact
- webhook / ingestion surface 只能消费现有 dashboard payload artifact

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/ci/eval_reporting_release_summary.json`
  - `reports/ci/eval_reporting_public_index.json`
  - `reports/ci/eval_reporting_dashboard_payload.json`
  - `reports/ci/eval_reporting_release_note_snippet.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
  - `Eval Reporting` GitHub status check
  - dashboard payload surface
  - release-note snippet surface
- 当前仍缺：
  - 一个专门给 release draft / release body 预填充消费的 canonical artifact
  - 一个给 external dashboard webhook / ingestion 复用的 export surface

---

## Batch 11A：Release Draft Prefill Surface

### 目标

生成一个给 release manager、release draft、handoff 文本直接复用的最小 prefill artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_release_draft_prefill.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_release_draft_prefill.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_note_snippet.json`
- 归一化为 release draft / release body 友好的最小 prefill

它不允许负责：

- 再次读取 dashboard payload / release summary / public index / stack summary
- 重新生成 snippet
- 直接创建 GitHub release / draft release
- 修改 notify / PR comment / status check owner

#### artifact 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_prefill.json`
- `reports/ci/eval_reporting_release_draft_prefill.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_prefill"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `source_snippet_surface_kind`

其中：

- `draft_title` 只能由现有 snippet 的 `release_readiness` / `headline` 薄推导得到
- `draft_body_markdown` 必须能直接用于 release draft body 或人工 handoff

Markdown 至少要包含：

- `Eval Reporting`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft prefill`
- always-run `Append eval reporting release draft prefill to job summary`
- always-run `Upload eval reporting release draft prefill`

这些步骤必须位于 release-note snippet 之后。

### Batch 11A 验收条件

必须同时满足：

- release-draft prefill 只消费 release-note snippet artifact
- prefill artifact 能稳定 materialize
- job summary 中出现 release-draft / handoff 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 11B：External Dashboard Webhook / Ingestion Export Surface

### 目标

让 `evaluation-report.yml` 生成一个稳定的 external dashboard webhook / ingestion export surface，给后续 webhook / third-party ingestion 直接复用。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_webhook_export.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_webhook_export.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_dashboard_payload.json`
- 归一化为 webhook / ingestion 友好的最小 export payload

它不允许负责：

- 再次读取 release snippet / release summary / public index / stack summary
- 重新生成 dashboard payload
- 真正发出 HTTP 请求
- 新增 dashboard server / webhook sender

#### export 合同

建议默认输出：

- `reports/ci/eval_reporting_webhook_export.json`
- `reports/ci/eval_reporting_webhook_export.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_webhook_export"`
- `generated_at`
- `release_readiness`
- `stack_status`
- `dashboard_headline`
- `missing_count`
- `stale_count`
- `mismatch_count`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `webhook_event_type`
- `ingestion_schema_version`

其中：

- `webhook_event_type` 建议固定为稳定值，例如 `eval_reporting.updated`
- `ingestion_schema_version` 只能作为 thin export version，不能变成新的 owner schema

Markdown 至少要包含：

- `Webhook Event`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting webhook export`
- always-run `Append eval reporting webhook export to job summary`
- always-run `Upload eval reporting webhook export`

这些步骤必须位于 dashboard payload 之后，且与 release-draft surface 并列，不要求先后依赖。

#### 不做的事

本批不做：

- 真正调用外部 webhook
- 真正发送 third-party ingestion 请求
- 自动创建 GitHub release
- 修改 notify / PR comment / Pages owner

### Batch 11B 验收条件

必须同时满足：

- webhook export 只消费 dashboard payload artifact
- workflow 中存在独立的 export 生成 / summary / upload steps
- export payload 可直接给后续 webhook sender / ingestion job 复用
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 11A

- 新增：
  - `tests/unit/test_generate_eval_reporting_release_draft_prefill.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 11B

- 新增：
  - `tests/unit/test_generate_eval_reporting_webhook_export.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 11A

- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PREFILL_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PREFILL_SURFACE_ALIGNMENT_VALIDATION_20260331.md`

### Batch 11B

- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_EXPORT_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_WEBHOOK_EXPORT_SURFACE_ALIGNMENT_VALIDATION_20260331.md`
