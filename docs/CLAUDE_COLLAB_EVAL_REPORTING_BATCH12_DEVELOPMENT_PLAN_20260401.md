# Claude Collaboration Batch 12 Development Plan

日期：2026-04-01

## 目标概览

本轮协作开发只做 `eval reporting` 的 GitHub release draft 自动化消费面，顺序固定：

1. 先生成一个给 GitHub release draft API / draft workflow 直接消费的 canonical payload artifact
2. 再基于这份 payload 生成 gated dry-run / optional publish surface

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_release_draft_prefill.py`
  - `scripts/ci/generate_eval_reporting_release_note_snippet.py`
  - `scripts/ci/generate_eval_reporting_dashboard_payload.py`
- 只能新增 thin release-draft-payload helper / thin release-draft-dry-run consumer
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary / dashboard payload / release-note snippet / release-draft prefill
- release-draft payload 只能消费现有 release-draft prefill artifact
- dry-run / optional publish surface 只能消费新的 release-draft payload artifact

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/ci/eval_reporting_release_draft_prefill.json`
  - `reports/ci/eval_reporting_webhook_export.json`
  - `reports/ci/eval_reporting_release_note_snippet.json`
  - `reports/ci/eval_reporting_dashboard_payload.json`
  - `reports/ci/eval_reporting_public_index.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
  - `Eval Reporting` GitHub status check
  - release-draft prefill surface
  - webhook export surface
- 当前仍缺：
  - 一个专门给 GitHub release draft API / draft workflow 消费的 canonical payload
  - 一个默认 dry-run、可 gated optional publish 的 release draft surface

---

## Batch 12A：GitHub Release Draft Payload Surface

### 目标

生成一个给 GitHub release draft、release manager、后续发布 automation 直接消费的最小 canonical payload。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_release_draft_payload.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_release_draft_payload.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_prefill.json`
- 归一化为 GitHub release draft 友好的最小 payload

它不允许负责：

- 再次读取 release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 release-draft prefill
- 直接创建或编辑 GitHub release
- 修改 notify / PR comment / status check owner

#### payload 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_payload.json`
- `reports/ci/eval_reporting_release_draft_payload.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_payload"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `repository_url`
- `source_prefill_surface_kind`

其中：

- `draft_title` 只能由现有 prefill 的 `draft_title` / `release_readiness` 薄推导得到
- `draft_body_markdown` 必须可直接用于 GitHub release draft body 或人工 handoff
- `repository_url` 只能由 workflow context 或现有已知 repo 信息薄拼接得到，不能引入新 owner

Markdown 至少要包含：

- `Eval Reporting`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft payload`
- always-run `Append eval reporting release draft payload to job summary`
- always-run `Upload eval reporting release draft payload`

这些步骤必须位于 release draft prefill 之后。

### Batch 12A 验收条件

必须同时满足：

- release-draft payload 只消费 release-draft prefill artifact
- payload artifact 能稳定 materialize
- job summary 中出现 release draft 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 12B：Gated Release Draft Dry-Run / Optional Publish Surface

### 目标

让 `evaluation-report.yml` 基于 release-draft payload 生成一个默认 dry-run、在明确 gate 下才 optional publish 的 GitHub release draft surface。

### 必做改动

1. 新增 `scripts/ci/post_eval_reporting_release_draft_dry_run.js` 或等价 thin consumer
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### consumer 职责

`post_eval_reporting_release_draft_dry_run.js` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_draft_payload.json`
- 归一化为 GitHub release draft dry-run / optional publish surface
- 在 gated 条件满足时，可选调用 GitHub API 创建 draft release

它不允许负责：

- 再次读取 release-draft prefill / release-note snippet / dashboard payload / release summary / public index / stack summary
- 重新生成 release-draft payload
- 在默认路径下自动创建 GitHub release
- 修改 notify / PR comment / status check owner

#### dry-run / publish 合同

建议默认输出：

- `reports/ci/eval_reporting_release_draft_dry_run.json`
- `reports/ci/eval_reporting_release_draft_dry_run.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_draft_dry_run"`
- `generated_at`
- `release_readiness`
- `draft_title`
- `draft_body_markdown`
- `publish_enabled`
- `publish_attempted`
- `publish_allowed`
- `publish_mode`
- `github_release_tag`

其中：

- 默认必须是 dry-run，不得在未授权上下文中自动发布
- 只有在显式 gated 条件满足时才允许 optional publish
- 无权限场景必须 fail-soft，不得破坏主评估链
- 如果使用 GitHub API，必须清晰区分 dry-run artifact 与真实 publish attempt

Markdown 至少要包含：

- `Dry Run`
- `Publish Enabled`
- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release draft dry run`
- always-run `Append eval reporting release draft dry run to job summary`
- always-run `Upload eval reporting release draft dry run`

这些步骤必须位于 release draft payload 之后，且默认不做实际发布。

#### 不做的事

本批不做：

- 自动发布 non-draft release
- 自动编辑 changelog
- 修改 notify / PR comment / Pages owner
- 接入 external webhook sender

### Batch 12B 验收条件

必须同时满足：

- dry-run surface 只消费 release-draft payload artifact
- workflow 中存在独立的 dry-run / optional publish steps
- default 路径不真正发布 GitHub release
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 12A

- 新增：
  - `tests/unit/test_generate_eval_reporting_release_draft_payload.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 12B

- 新增：
  - `tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 12A

- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PAYLOAD_SURFACE_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_PAYLOAD_SURFACE_ALIGNMENT_VALIDATION_20260401.md`

### Batch 12B

- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_DRY_RUN_SURFACE_ALIGNMENT_DESIGN_20260401.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_DRAFT_DRY_RUN_SURFACE_ALIGNMENT_VALIDATION_20260401.md`
