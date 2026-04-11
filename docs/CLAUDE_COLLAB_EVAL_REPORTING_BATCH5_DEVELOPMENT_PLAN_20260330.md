# Claude Collaboration Batch 5 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的 CI / cron 接线与运营化告警，顺序固定：

1. 先把 `.github/workflows/evaluation-report.yml` 接到默认的 top-level `eval-reporting-refresh`
2. 再补 workflow 级 summary / annotation / artifact summary surface

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/refresh_eval_reporting_stack.py`
  - `scripts/ci/generate_eval_reporting_bundle.py`
  - `scripts/ci/check_eval_reporting_bundle_health.py`
  - `scripts/ci/generate_eval_reporting_index.py`
- 只能新增 thin workflow consumer / thin summary helper
- 不允许新建新的 summary owner / metrics owner
- 不允许改动 `eval_signal` / `history_sequence` / top-level bundle schema 语义
- `static report` / `interactive report` / sub-bundle / top-level bundle 仍必须可独立运行

---

## 当前真实基线

截至当前仓库状态：

- `scripts/ci/refresh_eval_reporting_stack.py` 已存在，并按顺序 fail-closed：
  1. `generate_eval_reporting_bundle`
  2. `check_eval_reporting_bundle_health`
  3. `generate_eval_reporting_index`
- `scripts/ci/check_eval_reporting_bundle_health.py` 已输出：
  - `eval_reporting_bundle_health_report.json`
  - `eval_reporting_bundle_health_report.md`
- `scripts/ci/generate_eval_reporting_index.py` 已输出：
  - `eval_reporting_index.json`
  - `eval_reporting_index.md`
- `scripts/eval_with_history.sh` 已默认 materialize：
  1. `history_sequence_reporting_bundle`
  2. `eval_reporting_bundle`
  3. explainability guard
- `.github/workflows/evaluation-report.yml` 仍保留旧式分散步骤：
  - `Generate trend charts`
  - `Generate HTML report`
  - `Generate weekly rolling summary`
- `.github/workflows/evaluation-report.yml` 当前 `REPORT_PATH` 仍指向 `reports/eval_history/report`
- workflow 还没有默认调用 `eval-reporting-refresh`
- workflow 还没有上传 top-level stack 的 bundle / health / index artifact
- workflow 还没有基于 health report 的统一 summary / annotation surface

---

## Batch 5A：Workflow Wiring + Diagnostics Retention

### 目标

让 `.github/workflows/evaluation-report.yml` 成为 top-level eval reporting stack 的默认执行入口，并在失败时保留诊断 artifact。

### 必做改动

1. 修改 `.github/workflows/evaluation-report.yml`
2. 新增或更新 workflow regression tests

### 设计约束

#### workflow 必须直接接现有 refresh owner

workflow 只能调用现有：

- `make eval-reporting-refresh`
  或
- `python3 scripts/ci/refresh_eval_reporting_stack.py`

但不能：

- 在 workflow 里重新拼装 `generate_eval_reporting_bundle` / `check_eval_reporting_bundle_health` / `generate_eval_reporting_index`
- 再保留旧的分散生成步骤作为主路径
- 直接在 workflow 里重算 summary / trend / weekly

#### static / interactive 路径必须对齐 canonical output

workflow 环境变量必须显式对齐当前 top-level bundle 默认路径：

- `REPORT_PATH=reports/eval_history/report_static`
- `INTERACTIVE_REPORT_PATH=reports/eval_history/report_interactive`
- `EVAL_REPORTING_BUNDLE_JSON=reports/eval_history/eval_reporting_bundle.json`
- `EVAL_REPORTING_BUNDLE_HEALTH_JSON=reports/eval_history/eval_reporting_bundle_health_report.json`
- `EVAL_REPORTING_INDEX_JSON=reports/eval_history/eval_reporting_index.json`

如需新增 `EVAL_REPORTING_REFRESH_DAYS`，只能作为 thin workflow env，不得改动 owner schema。

#### refresh step 必须保留诊断再失败

本批 workflow 里的 refresh step 必须满足：

- 可以使用 `continue-on-error: true`
- 必须显式记录 `exit_code`
- 必须在 top-level artifact upload 之后再决定是否阻断 workflow

也就是说：

- 允许 refresh step 自身先不阻断，目的是保留 bundle / health / index / report artifact
- 但 workflow 最终仍必须 fail-closed
- 不允许像 Batch 4B 原始问题那样把失败吞掉

#### 上传 artifact 的范围

workflow 必须新增或调整上传步骤，至少包含：

- static report artifact
- interactive report artifact
- top-level eval reporting stack artifact

top-level stack artifact 至少应包含：

- `eval_reporting_bundle.json`
- `eval_reporting_bundle.md`
- `eval_reporting_bundle_health_report.json`
- `eval_reporting_bundle_health_report.md`
- `eval_reporting_index.json`
- `eval_reporting_index.md`

如 workflow 继续保留原 `evaluation-report-${{ github.run_number }}` artifact 名称，可以；但 static / interactive 不能再混到同一路径。

#### 不做的事

本批不做：

- 新建 workflow 文件
- 修改 PR comment JS contract
- 修改通知脚本 contract
- 新增 HTML landing page
- 新增新的 metrics owner

### Batch 5A 验收条件

必须同时满足：

- `evaluation-report.yml` 默认走 `eval-reporting-refresh`
- workflow 不再以旧的 `Generate trend charts / Generate HTML report / Generate weekly rolling summary` 作为主生成路径
- static / interactive report 路径与 top-level bundle 默认路径对齐
- refresh failure 不会被吞掉，但 diagnostics artifact 仍会保留
- top-level bundle / health / index artifact 会被上传

---

## Batch 5B：Workflow Summary + Alert Surface

### 目标

让 workflow 对 top-level eval reporting stack 的健康状态有统一 summary / annotation surface，而不是只靠 job fail/red。

### 必做改动

1. 新增一个 thin helper
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增或更新 workflow regression tests

### 设计约束

#### 允许新增的 thin helper

推荐新增：

- `scripts/ci/summarize_eval_reporting_stack_status.py`

它只能负责：

- 读取：
  - `eval_reporting_bundle.json`
  - `eval_reporting_bundle_health_report.json`
  - `eval_reporting_index.json`
- 归一化成 workflow-friendly summary
- 输出 JSON / Markdown summary artifact

它不允许负责：

- materialize bundle
- 重算 summary / weekly / trend
- HTML render
- 新 owner schema

#### summary artifact 合同

建议默认输出：

- `reports/ci/eval_reporting_stack_summary.json`
- `reports/ci/eval_reporting_stack_summary.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_stack_summary"`
- `generated_at`
- `refresh_exit_code`
- `bundle_status`
- `health_status`
- `index_status`
- `missing_count`
- `stale_count`
- `mismatch_count`
- `static_report_html`
- `interactive_report_html`
- `eval_signal_bundle_json`
- `history_sequence_bundle_json`

#### workflow 级展示

workflow 必须新增：

- 一个 always-run summary step
- 一个 always-run summary artifact upload step
- 一个 append-to-`$GITHUB_STEP_SUMMARY` step

允许新增 annotation step，但只能消费 health / summary artifact。

#### fail 顺序

本批结束后 workflow 顺序必须是：

1. refresh step
2. summary/annotation/artifact upload
3. 最终 fail step

不允许在 summary / artifact upload 之前就提前 fail。

#### 不做的事

本批不做：

- PR comment JS 扩展
- Slack / SMTP payload 扩展
- GitHub Pages landing page
- 新 compare / trend 图表

### Batch 5B 验收条件

必须同时满足：

- workflow 能 materialize `eval_reporting_stack_summary.json/md`
- summary / annotation 只消费现有 bundle / health / index
- job summary 会显示 top-level eval reporting stack 状态
- upload / summary 完成后 workflow 仍会按 refresh exit code fail-closed

---

## 必须新增或更新的测试

### Batch 5A

- 新增：
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
- 更新或重跑：
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - `tests/unit/test_generate_eval_reporting_bundle.py`
  - `tests/unit/test_check_eval_reporting_bundle_health.py`
  - `tests/unit/test_generate_eval_reporting_index.py`
  - `tests/unit/test_refresh_eval_reporting_stack.py`

### Batch 5B

- 新增：
  - `tests/unit/test_summarize_eval_reporting_stack_status.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
- 重跑：
  - Batch 5A 全部测试

---

## 本轮必须新增的设计 / 验证 MD

### Batch 5A

- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_REFRESH_WIRING_AND_DIAGNOSTIC_RETENTION_ALIGNMENT_VALIDATION_20260330.md`

### Batch 5B

- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVALUATION_REPORT_WORKFLOW_STACK_SUMMARY_AND_ALERT_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

---

## Deferred（本轮明确不做）

- 顶层 discovery HTML landing page
- PR comment 对 eval reporting stack 的富展示
- Slack / 邮件通知正文扩展
- 新的 report/trend/leaderboard
- 任何新的 metrics owner / summary owner
