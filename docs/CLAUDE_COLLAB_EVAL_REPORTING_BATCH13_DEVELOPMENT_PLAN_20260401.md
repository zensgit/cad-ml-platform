# Claude Collaboration Batch 13 Development Plan

日期：2026-04-01

## 目标概览

本轮协作开发只做 `eval reporting` 的 GitHub release draft 发布自动化面，顺序固定：

1. 先生成一个给 gated GitHub release draft publish 直接消费的 canonical publish payload / policy artifact
2. 再基于这份 payload 实现默认关闭、显式 gate 才尝试执行的 draft publish automation

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_release_draft_payload.py`
  - `scripts/ci/post_eval_reporting_release_draft_dry_run.js`
  - `scripts/ci/generate_eval_reporting_release_draft_prefill.py`
- 只能新增 thin publish-payload helper / thin publish consumer
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary / dashboard payload / release-note snippet / release-draft prefill / release-draft payload
- publish payload 只能消费现有 release-draft payload artifact
- publish automation 只能消费新的 publish payload artifact

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/ci/eval_reporting_release_draft_prefill.json`
  - `reports/ci/eval_reporting_release_draft_payload.json`
  - `reports/ci/eval_reporting_release_draft_dry_run.json`
  - `reports/ci/eval_reporting_webhook_export.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
  - `Eval Reporting` GitHub status check
  - release-draft prefill surface
  - release-draft payload surface
  - release-draft dry-run surface
- 当前仍缺：
  - 一个专门给 gated GitHub release draft publish 消费的 canonical publish payload / policy artifact
  - 一个真正的 optional GitHub release draft publish automation surface

---

## Batch 13A：Release Draft Publish Payload / Policy Surface

### 目标

生成一个给 GitHub release draft publish automation、release manager、后续发布 gate 直接消费的最小 canonical payload / policy artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_release_draft_publish_payload.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_release_draft_publish_payload.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_payload.json`
- 归一化为 GitHub release draft publish 友好的最小 payload / policy

它不允许负责：

- 再次读取 release-draft prefill / release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 release-draft payload
- 直接创建或编辑 GitHub release
- 修改 notify / PR comment / status check owner

#### payload / policy 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_publish_payload.json`
- `reports/ci/eval_reporting_release_draft_publish_payload.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_publish_payload"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `github_release_tag`
- `publish_policy`
- `publish_allowed`
- `publish_requires_explicit_enable`
- `repository_url`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `source_release_draft_payload_surface_kind`

其中：

- `publish_policy` 只能是 thin policy，例如：
  - `disabled_by_default`
  - `ready_only`
  - `draft_release_only`
- `publish_allowed` 只能由现有 release readiness 和既定 gate 薄推导得到
- `publish_requires_explicit_enable` 必须固定表达“默认不发布”

Markdown 至少要包含：

- `Eval Reporting`
- `Release readiness`
- `Publish Policy`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft publish payload`
- always-run `Append eval reporting release draft publish payload to job summary`
- always-run `Upload eval reporting release draft publish payload`

这些步骤必须位于 release draft dry run 之后，或者至少位于 release draft payload 之后且在真正 publish step 之前。

### Batch 13A 验收条件

必须同时满足：

- publish payload 只消费 release-draft payload artifact
- payload / policy artifact 能稳定 materialize
- job summary 中出现 publish policy 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 13B：Optional GitHub Release Draft Publish Automation

### 目标

让 `evaluation-report.yml` 基于 publish payload 实现一个默认关闭、显式 gate 才尝试执行的 GitHub release draft publish automation。

### 必做改动

1. 新增 `scripts/ci/post_eval_reporting_release_draft_publish.js` 或等价 thin consumer
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### consumer 职责

`post_eval_reporting_release_draft_publish.js` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_publish_payload.json`
- 执行 gated publish decision
- 在显式 enable 且 gate 允许时，可选调用 GitHub API 创建 draft release

它不允许负责：

- 再次读取 release-draft payload / release-draft prefill / release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 publish payload
- 创建 non-draft release
- 修改 notify / PR comment / status check owner

#### publish automation 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_publish_result.json`
- `reports/ci/eval_reporting_release_draft_publish_result.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_publish_result"`
- `generated_at`
- `release_readiness`
- `publish_enabled`
- `publish_allowed`
- `publish_attempted`
- `publish_succeeded`
- `publish_mode`
- `github_release_tag`
- `github_release_id`

其中：

- 默认必须 `publish_enabled = false`
- 只有在显式 gate 满足时才允许尝试 publish
- 无权限或 API 失败场景必须 fail-soft，不得破坏主评估链
- 若 publish 未尝试或失败，artifact 仍必须稳定落盘

Markdown 至少要包含：

- `Publish Attempted`
- `Publish Succeeded`
- `Publish Mode`
- `Release readiness`
- `GitHub Release Tag`

#### workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft publish result`
- always-run `Append eval reporting release draft publish result to job summary`
- always-run `Upload eval reporting release draft publish result`

这些步骤必须位于 publish payload 之后，且默认不做实际发布。

#### 不做的事

本批不做：

- 自动发布 non-draft release
- 自动编辑 changelog
- 修改 notify / PR comment / Pages owner
- 接入 external webhook sender / retry queue

### Batch 13B 验收条件

必须同时满足：

- publish automation 只消费 publish payload artifact
- workflow 中存在独立的 publish result / optional publish steps
- default 路径不真正发布 GitHub release
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 13A

- 新增：
  - `tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 13B

- 新增：
  - `tests/unit/test_post_eval_reporting_release_draft_publish_js.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 13A

- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_PAYLOAD_ALIGNMENT_VALIDATION_20260401.md`

### Batch 13B

- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PUBLISH_AUTOMATION_ALIGNMENT_VALIDATION_20260401.md`
