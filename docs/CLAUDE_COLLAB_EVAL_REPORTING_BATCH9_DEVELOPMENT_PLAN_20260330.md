# Claude Collaboration Batch 9 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的 release / status check 外层消费面，顺序固定：

1. 先生成一个 release/status 友好的 canonical summary artifact
2. 再让 `evaluation-report.yml` 用这份 summary 发独立的 GitHub status check

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/summarize_eval_reporting_stack_status.py`
  - `scripts/ci/generate_eval_reporting_index.py`
  - `scripts/ci/generate_eval_reporting_public_index.py`
- 只能新增 thin release-summary helper / thin status-check helper
- 不允许新建新的 metrics owner
- 不允许重算 bundle / health / index / public index
- status check 只能消费新的 release summary artifact，不允许内联重算

---

## 当前真实基线

截至当前仓库状态：

- `evaluation-report.yml` 已稳定产出：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/ci/eval_reporting_public_index.json`（仅 deploy-pages job）
  - `reports/eval_history/index.html`
- `evaluation-report.yml` 已有：
  - PR comment surface
  - notify surface
  - Pages/public discovery surface
- 当前仍缺：
  - 一个专门给 release / status check 消费的 canonical summary artifact
  - 一个独立的 `Eval Reporting` GitHub status check / check-run surface
- 现有最接近的参考模式是：
  - `.github/workflows/release-risk-check.yml` 里的 status check step
  - 但 eval reporting 还没有自己的 status check surface

---

## Batch 9A：Release Summary Artifact

### 目标

生成一个给 release / status check 消费的最小 canonical summary artifact。

### 必做改动

1. 新增 `scripts/ci/generate_eval_reporting_release_summary.py`
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### helper 职责

`generate_eval_reporting_release_summary.py` 只能负责：

- 读取：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
- 归一化为 release/status 友好的最小 summary

它不允许负责：

- 重新生成 stack summary
- 重新生成 index / public index
- 重新渲染 landing page / reports
- 新建 metrics schema

#### summary 合同

建议默认输出：

- `reports/ci/eval_reporting_release_summary.json`
- `reports/ci/eval_reporting_release_summary.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_release_summary"`
- `generated_at`
- `stack_summary_status`
- `missing_count`
- `stale_count`
- `mismatch_count`
- `landing_page_path`
- `static_report_path`
- `interactive_report_path`
- `release_readiness`

其中：

- `release_readiness` 可以是 thin derived signal，例如：
  - `ready`
  - `degraded`
  - `unavailable`

但只能基于已有 stack summary / index 推导，不允许引入新门限。

#### workflow 改动约束

`evaluation-report.yml` 必须在 `evaluate` job 中新增：

- always-run `Generate eval reporting release summary`
- always-run `Append eval reporting release summary to job summary`
- always-run `Upload eval reporting release summary`

这些步骤必须位于 stack summary 之后，且在任何新的 status check step 之前。

### Batch 9A 验收条件

必须同时满足：

- release summary artifact 能稳定 materialize
- summary 只消费现有 index / stack summary
- job summary 中出现 release/status 友好的最小摘要
- artifact 已作为独立 surface 上传

---

## Batch 9B：GitHub Status Check Consumer

### 目标

让 `evaluation-report.yml` 基于 release summary 发一个独立的 `Eval Reporting` GitHub status check / check-run。

### 必做改动

1. 新增一个 thin JS helper 或最小 github-script consumer
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新测试

### 设计约束

#### 允许的实现方式

优先推荐：

- `scripts/ci/post_eval_reporting_status_check.js`

它只能负责：

- 读取 release summary JSON
- 归一化为 GitHub status check / check-run payload
- 发起 GitHub API 调用

它不允许负责：

- 重新读取 index / stack summary 再做第二次归一化
- 重新计算 release summary
- 生成 comment / notify / Pages surface

#### status check 合同

建议 check 名称固定为：

- `Eval Reporting`

建议状态映射：

- `release_readiness == "ready"` -> success
- `release_readiness == "degraded"` -> neutral
- `release_readiness == "unavailable"` -> failure

如果采用 commit status API 而不是 checks API，必须处理 GitHub state 约束，不能发无效 state。

#### workflow 约束

status check step 必须：

- 位于 release summary 生成之后
- 只在权限允许的上下文运行
- 在权限不足或 forks 等场景下 fail-soft 记录 warning，不得破坏主评估链

允许仿照 `release-risk-check.yml` 的 try/catch 模式。

#### 不做的事

本批不做：

- 跨 workflow 消费 release summary
- 修改 release-risk-check workflow
- 修改 PR comment / notify / Pages owner
- 新增 dashboard ingestion

### Batch 9B 验收条件

必须同时满足：

- status check 只消费 release summary artifact
- workflow 中存在独立 `Eval Reporting` status check step
- 无权限时 fail-soft，不破坏主工作流
- 新增测试明确覆盖 workflow wiring 与 payload 映射

---

## 必须新增或更新的测试

### Batch 9A

- 新增：
  - `tests/unit/test_generate_eval_reporting_release_summary.py`
  - `tests/unit/test_evaluation_report_workflow_release_summary.py`

### Batch 9B

- 新增：
  - `tests/unit/test_post_eval_reporting_status_check_js.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_release_summary.py`

---

## 建议的设计 / 验证 MD

### Batch 9A

- `docs/DEDUP_EVAL_REPORTING_RELEASE_SUMMARY_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_RELEASE_SUMMARY_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

### Batch 9B

- `docs/DEDUP_EVAL_REPORTING_STATUS_CHECK_CONSUMER_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_STATUS_CHECK_CONSUMER_ALIGNMENT_VALIDATION_20260330.md`

