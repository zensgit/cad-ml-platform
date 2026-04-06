# Claude Collaboration Batch 10 Development Plan

日期：2026-03-31

## 目标概览

本轮协作开发只做 `eval reporting` 的更外层消费面，顺序固定：

1. 先生成一个给 external dashboard / downstream ingestion 消费的 canonical payload artifact
2. 再基于这份 payload 生成 release-note 友好的 snippet artifact

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/summarize_eval_reporting_stack_status.py`
  - `scripts/ci/generate_eval_reporting_release_summary.py`
  - `scripts/ci/generate_eval_reporting_public_index.py`
- 只能新增 thin dashboard-payload helper / thin release-note helper
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index / release summary
- release-note surface 只能消费新的 dashboard payload artifact，不允许再次读取多份上游 JSON 自行拼装

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/ci/eval_reporting_release_summary.json`
  - `reports/ci/eval_reporting_public_index.json`
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
  - `Eval Reporting` GitHub status check
- 当前仍缺：
  - 一个专门给 external dashboard / ingestion 消费的稳定 canonical payload
  - 一个给 release note / human handoff 消费的最小 snippet artifact

---

## Batch 10A：Dashboard Payload Artifact

### 目标

生成一个给 external dashboard、后续 dashboard ingestion、以及其他外部消费者复用的最小 canonical payload。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_dashboard_payload.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_dashboard_payload.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_release_summary.json`
  - `reports/ci/eval_reporting_public_index.json`
- 归一化为 external dashboard 友好的最小 payload

它不允许负责：

- 重新生成 stack summary / release summary / public index
- 重新渲染 landing page / reports
- 新建 metrics schema

#### payload 合同

建议默认输出：

- `reports/ci/eval_reporting_dashboard_payload.json`
- `reports/ci/eval_reporting_dashboard_payload.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_dashboard_payload"`
- `generated_at`
- `release_readiness`
- `stack_status`
- `missing_count`
- `stale_count`
- `mismatch_count`
- `public_landing_page_url`
- `public_static_report_url`
- `public_interactive_report_url`
- `dashboard_headline`
- `public_discovery_ready`

其中：

- `dashboard_headline` 只能由现有 release summary / public index 薄推导得到
- `public_discovery_ready` 只能是基于现有 public URLs 是否完整的 thin signal

#### workflow 改动约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting dashboard payload`
- always-run `Append eval reporting dashboard payload to job summary`
- always-run `Upload eval reporting dashboard payload`

这些步骤必须位于 public index 生成之后，且在任何新的 release-note step 之前。

### Batch 10A 验收条件

必须同时满足：

- dashboard payload artifact 能稳定 materialize
- payload 只消费现有 release summary / public index
- job summary 中出现 dashboard / public URL 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 10B：Release Note Snippet Surface

### 目标

让 `evaluation-report.yml` 基于 dashboard payload 生成一个 release-note / handoff 友好的最小 snippet surface。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_release_note_snippet.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_release_note_snippet.py` 只能负责：

- 读取：
  - `reports/ci/eval_reporting_dashboard_payload.json`
- 输出 release-note / handoff 友好的 snippet

它不允许负责：

- 再次读取 release summary / public index / stack summary
- 重新生成 payload
- 直接发布 release note / GitHub release
- 修改 PR comment / notify / status check owner

#### snippet 合同

建议默认输出：

- `reports/ci/eval_reporting_release_note_snippet.md`
- `reports/ci/eval_reporting_release_note_snippet.json`

其中 JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_note_snippet"`
- `generated_at`
- `release_readiness`
- `headline`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `snippet_markdown`

Markdown 至少要包含：

- `Release readiness`
- `Landing Page`
- `Static Report`
- `Interactive Report`

#### workflow 约束

`evaluation-report.yml` 必须在 `deploy-pages` job 中新增：

- always-run `Generate eval reporting release note snippet`
- always-run `Append eval reporting release note snippet to job summary`
- always-run `Upload eval reporting release note snippet`

这些步骤必须位于 dashboard payload 之后。

#### 不做的事

本批不做：

- 自动创建 GitHub release
- 自动编辑 release notes / changelog
- 修改 notify / PR comment / Pages owner
- 新增 dashboard server / ingestion backend

### Batch 10B 验收条件

必须同时满足：

- release-note snippet 只消费 dashboard payload artifact
- workflow 中存在独立的 snippet 生成 / summary / upload steps
- snippet 可直接被人工复制到 release note / handoff 文本中
- 新增测试明确覆盖 workflow wiring 与 artifact 合同

---

## 必须新增或更新的测试

### Batch 10A

- 新增：
  - `tests/unit/test_generate_eval_reporting_dashboard_payload.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

### Batch 10B

- 新增：
  - `tests/unit/test_generate_eval_reporting_release_note_snippet.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 10A

- `docs/DEDUP_EVAL_REPORTING_DASHBOARD_PAYLOAD_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_DASHBOARD_PAYLOAD_SURFACE_ALIGNMENT_VALIDATION_20260331.md`

### Batch 10B

- `docs/DEDUP_EVAL_REPORTING_RELEASE_NOTE_SNIPPET_SURFACE_ALIGNMENT_DESIGN_20260331.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_NOTE_SNIPPET_SURFACE_ALIGNMENT_VALIDATION_20260331.md`
