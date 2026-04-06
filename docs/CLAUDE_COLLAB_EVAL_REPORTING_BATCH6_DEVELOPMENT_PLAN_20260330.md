# Claude Collaboration Batch 6 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的顶层人类入口收口，顺序固定：

1. 先补一个轻量 HTML landing / discovery page
2. 再把 landing page 接进现有 refresh / artifact / Pages 交付链

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/generate_eval_reporting_bundle.py`
  - `scripts/ci/check_eval_reporting_bundle_health.py`
  - `scripts/ci/generate_eval_reporting_index.py`
  - `scripts/ci/summarize_eval_reporting_stack_status.py`
- 只能新增 thin loader / thin renderer / thin workflow wiring
- 不允许新建新的 summary owner / metrics owner
- landing page 只做 discovery / navigation / status overview
- landing page 不复制 detailed metrics 表格，不接管 report HTML render

---

## 当前真实基线

截至当前仓库状态：

- `eval_signal` / `history_sequence` / `top-level eval reporting` 三层 owner/wrapper 已稳定
- `scripts/ci/refresh_eval_reporting_stack.py` 已作为默认 refresh 入口
- `scripts/ci/check_eval_reporting_bundle_health.py` 已输出 health report
- `scripts/ci/generate_eval_reporting_index.py` 已输出 discovery/index JSON/MD
- `scripts/ci/summarize_eval_reporting_stack_status.py` 已输出 workflow-friendly stack summary
- `.github/workflows/evaluation-report.yml` 已：
  - 默认调用 refresh
  - 上传 static / interactive / top-level stack artifact
  - 上传 stack summary artifact
  - 在 `$GITHUB_STEP_SUMMARY` 追加 stack summary
- 当前缺口：
  - 没有一个给人直接看的顶层 HTML landing page
  - static / interactive report 仍然需要人工猜路径
  - Pages / artifact 交付还没有统一把 landing page 作为入口页

---

## Batch 6A：Landing Page Renderer

### 目标

新增一个极薄的 HTML landing / discovery page，统一展示 top-level eval reporting stack 的入口链接和状态。

### 必做改动

1. 新增 `scripts/generate_eval_reporting_landing_page.py`
2. 如有必要，新增一个很薄的 loader/helper
3. 新增或更新测试

### 设计约束

#### 允许新增的 thin helper

如有必要，推荐新增：

- `scripts/eval_reporting_landing_helpers.py`

它只能负责：

- 读取：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `reports/eval_history/eval_reporting_bundle_health_report.json`
- 归一化 discovery / status context

它不允许负责：

- 生成 summary metrics
- materialize bundle / health / index
- trend plotting
- Pages / workflow orchestration

#### landing page 输入

renderer 默认必须从现有 canonical artifacts 读取：

- `eval_reporting_index.json`
- `eval_reporting_stack_summary.json`
- `eval_reporting_bundle_health_report.json`

可以 fallback 缺失情况，但要明确显示“artifact missing”，不能悄悄静默吞掉。

#### landing page 输出

建议默认输出：

- `reports/eval_history/index.html`

可选支持：

- `--out`
- `--eval-history-dir`
- `--stack-summary-json`
- `--index-json`
- `--health-json`

#### 页面内容

landing page 只能展示：

- stack status
- refresh / health summary counts
- static report link
- interactive report link
- eval-signal bundle link
- history-sequence bundle link
- top-level bundle link
- health report link
- stack summary link

允许有小的状态 badge / list / table。

不允许：

- 重复渲染完整 static report 内容
- 重复渲染 interactive chart
- 重新计算 metrics
- 新增 compare leaderboard

#### HTML/CSS 约束

- 单文件静态 HTML，优先内联最小 CSS
- 不引入新的 JS 依赖
- 缺失 artifact 时仍能渲染页面
- 页面必须清楚区分：
  - `ok`
  - `degraded`
  - `missing`

### Batch 6A 验收条件

必须同时满足：

- landing page 可在只依赖现有 index/summary/health artifact 的情况下生成
- 页面可直接链接到 static / interactive / bundle / health / summary
- 缺失 artifact 时页面仍可生成并显示 missing 状态
- thin helper 若存在，没有越权成为新的 owner

---

## Batch 6B：Landing Integration

### 目标

把 landing page 接进现有 refresh / artifact / Pages 交付链，成为默认人类入口。

### 必做改动

1. 修改 `scripts/ci/refresh_eval_reporting_stack.py`
2. 视需要最小扩充 `scripts/ci/generate_eval_reporting_index.py`
3. 修改 `.github/workflows/evaluation-report.yml`
4. 更新 `Makefile`

### 设计约束

#### refresh integration

`scripts/ci/refresh_eval_reporting_stack.py` 可以新增第 4 步：

4. `generate_eval_reporting_landing_page.py`

但必须保持：

- fail-closed
- 仍只做 orchestrator
- 不新增新的 summary / metrics owner

#### index / bundle additive fields

本批允许增量补充字段，但不能改旧字段语义。

优先考虑只在 `eval_reporting_index.json` 里新增：

- `landing_page_html`

如确有必要，也可以在 top-level bundle 里增量补：

- `landing_page_html`

但必须是 additive，不得改已有 path contract。

#### workflow integration

`.github/workflows/evaluation-report.yml` 必须：

- 上传 landing page artifact
- 如果 Pages 仍只发布单一 artifact，则要把 landing page 作为入口根页
- 保持 static / interactive report 原 artifact 兼容

可接受两种实现：

1. 让 Pages artifact 根目录就是 `reports/eval_history`
2. 或生成一个单独的 Pages-ready root，再明确包含：
   - `index.html`
   - `report_static/`
   - `report_interactive/`

但必须明确，不允许靠隐式相对路径碰运气。

#### Make target

新增：

- `eval-reporting-landing-page`

并视需要更新：

- `eval-reporting-refresh`

仍必须保持 thin wrapper。

#### 不做的事

本批不做：

- PR comment 集成
- Slack / 邮件通知扩展
- 新 compare / trend 图
- 新 summary owner

### Batch 6B 验收条件

必须同时满足：

- refresh 能默认 materialize landing page
- index 或 bundle 能稳定暴露 landing page path
- workflow / artifact / Pages 至少有一条正式交付链以 landing page 为入口
- 现有 static / interactive report 兼容不被破坏

---

## 必须新增或更新的测试

### Batch 6A

- 新增：
  - `tests/unit/test_generate_eval_reporting_landing_page.py`
- 如新增 helper，再新增：
  - `tests/unit/test_eval_reporting_landing_helpers.py`
- 重跑：
  - `tests/unit/test_generate_eval_reporting_index.py`
  - `tests/unit/test_summarize_eval_reporting_stack_status.py`

### Batch 6B

- 更新：
  - `tests/unit/test_refresh_eval_reporting_stack.py`
  - `tests/unit/test_generate_eval_reporting_index.py`
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`
  - `tests/unit/test_eval_history_make_targets.py`
- 如 workflow / Pages 路径发生变化，新增针对 workflow artifact / deploy contract 的测试

---

## 本轮必须新增的设计 / 验证 MD

### Batch 6A

- `docs/DEDUP_EVAL_REPORTING_LANDING_PAGE_DISCOVERY_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_LANDING_PAGE_DISCOVERY_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

### Batch 6B

- `docs/DEDUP_EVAL_REPORTING_LANDING_PAGE_REFRESH_AND_DELIVERY_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_LANDING_PAGE_REFRESH_AND_DELIVERY_ALIGNMENT_VALIDATION_20260330.md`

---

## Deferred（本轮明确不做）

- PR comment 消费 top-level stack summary
- Slack / 邮件通知消费 top-level stack summary
- 新 compare / trend / leaderboard
- 任何新的 metrics owner / summary owner
