# Claude Collaboration Batch 8 Development Plan

日期：2026-03-30

## 目标概览

本轮协作开发只做 `eval reporting` 的 GitHub Pages / external discovery surface 收口，顺序固定：

1. 先把 landing page / static report / interactive report 收成一个显式的 Pages-ready root
2. 再补 deployment 后的 public discovery surface

执行原则：

- 必须沿用现有 canonical owner：
  - `scripts/ci/refresh_eval_reporting_stack.py`
  - `scripts/ci/generate_eval_reporting_bundle.py`
  - `scripts/ci/check_eval_reporting_bundle_health.py`
  - `scripts/ci/generate_eval_reporting_index.py`
  - `scripts/generate_eval_reporting_landing_page.py`
- 只能新增 thin assembler / thin deploy-summary helper
- 不允许新建新的 summary owner / metrics owner
- 不允许重算 bundle / health / index / landing 内容
- GitHub Pages 相关改动只能围绕“如何发布现有 canonical artifacts”展开

---

## 当前真实基线

截至当前仓库状态：

- `eval reporting` stack 已稳定产出：
  - `reports/eval_history/index.html`
  - `reports/eval_history/report_static/index.html`
  - `reports/eval_history/report_interactive/index.html`
  - `reports/eval_history/eval_reporting_bundle.json`
  - `reports/eval_history/eval_reporting_bundle_health_report.json`
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
- `.github/workflows/evaluation-report.yml` 的 `evaluate` job 已上传：
  - static report artifact
  - interactive report artifact
  - top-level eval reporting stack artifact
  - landing page artifact
- `.github/workflows/evaluation-report.yml` 的 `deploy-pages` job 当前仍只是：
  - 下载 `evaluation-report-${{ github.run_number }}`
  - 放到 `./public`
  - 上传到 Pages
- 当前缺口：
  - Pages publish root 仍默认是 static report artifact，不是 landing/discovery root
  - Pages job 没有显式消费一个 “landing page + static + interactive” 的 canonical pages-ready root
  - 部署后没有独立的 public discovery manifest / summary，外部 URL 只存在于 Pages deployment step output

---

## Batch 8A：Pages-Ready Root Alignment

### 目标

把 GitHub Pages 发布入口显式收口成 `landing page = public root` 的 canonical pages-ready root。

### 必做改动

1. 新增一个 thin assembler
2. 修改 `.github/workflows/evaluation-report.yml`
3. 新增/更新 workflow regression tests

### 设计约束

#### 允许新增的 thin assembler

推荐新增：

- `scripts/ci/assemble_eval_reporting_pages_root.py`

它只能负责：

- 读取现有已生成文件：
  - `reports/eval_history/index.html`
  - `reports/eval_history/report_static/`
  - `reports/eval_history/report_interactive/`
  - 如有需要，再附带 canonical JSON/MD artifacts
- 组装成一个 Pages-ready root，例如：
  - `reports/eval_pages/index.html`
  - `reports/eval_pages/report_static/...`
  - `reports/eval_pages/report_interactive/...`

它不允许负责：

- 重新生成 landing page
- 重新生成 static / interactive report
- 重算 index / bundle / health / summary
- 新 owner schema

#### Pages-ready root 合同

组装后的 root 必须满足：

- 根页必须是 landing page
- `report_static/` 和 `report_interactive/` 作为子目录保留
- 如附带 canonical JSON/MD，只能是 additive discovery asset

推荐至少包含：

- `index.html`
- `report_static/`
- `report_interactive/`
- `eval_reporting_bundle.json`
- `eval_reporting_bundle_health_report.json`
- `eval_reporting_index.json`

#### workflow 改动约束

`evaluate` job 必须新增 Pages-ready artifact 上传，例如：

- `eval-reporting-pages-${{ github.run_number }}`

`deploy-pages` job 必须改为下载这个 artifact，而不是继续下载 `evaluation-report-${{ github.run_number }}`

不允许：

- 继续让 Pages root 指向 static report artifact
- 在 `deploy-pages` job 里重新拼装 root
- 用 shell 直接复制一堆文件而不形成明确 assembler owner

### Batch 8A 验收条件

必须同时满足：

- Pages 发布根目录明确由 landing page 驱动
- deploy-pages job 下载 dedicated pages artifact
- Pages artifact 明确包含 landing page + static + interactive
- assembler 是 thin wrapper，不越权成新的 owner

---

## Batch 8B：Public Discovery Surface

### 目标

在 Pages deploy 成功后，为外部消费面补一层最小 public discovery surface。

### 必做改动

1. 新增一个 thin public-surface helper 或 script
2. 修改 `.github/workflows/evaluation-report.yml` 的 `deploy-pages` job
3. 新增/更新 workflow regression tests

### 设计约束

#### 允许新增的 thin helper

推荐新增：

- `scripts/ci/generate_eval_reporting_public_index.py`

它只能负责：

- 读取：
  - `reports/eval_history/eval_reporting_index.json`
  - `reports/ci/eval_reporting_stack_summary.json`
  - `page_url` / Pages base URL
- 生成 public-facing discovery surface

它不允许负责：

- 重新生成 landing page
- 重算 stack summary
- 重算 health / bundle / index
- 新 owner schema

#### public surface 合同

建议默认输出：

- `reports/ci/eval_reporting_public_index.json`
- `reports/ci/eval_reporting_public_index.md`

JSON 至少包含：

- `status`
- `surface_kind = "eval_reporting_public_index"`
- `generated_at`
- `page_url`
- `landing_page_url`
- `static_report_url`
- `interactive_report_url`
- `stack_summary_status`
- `missing_count`
- `stale_count`
- `mismatch_count`

Markdown 只做 discovery，不复制大表。

#### workflow / summary 约束

`deploy-pages` job 可以新增：

- always-run public index generation step
- upload public index artifact
- append-to-job-summary step

但不允许：

- 改动 deploy-pages 本身的 deploy owner
- 新增对现有 pages deployment 的阻断条件
- 重写 Pages URL 逻辑

### Batch 8B 验收条件

必须同时满足：

- deploy-pages 成功后能 materialize public discovery artifact
- public discovery 只消费现有 index / stack summary / Pages output URL
- job summary 能直接看到 landing/static/interactive 的 public URL
- 不引入新的 metrics owner

---

## 必须新增或更新的测试

### Batch 8A

- 新增：
  - `tests/unit/test_assemble_eval_reporting_pages_root.py`
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_eval_reporting_stack.py`

### Batch 8B

- 新增：
  - `tests/unit/test_generate_eval_reporting_public_index.py`
- 更新：
  - `tests/unit/test_evaluation_report_workflow_pages_deploy.py`

---

## 建议的设计 / 验证 MD

### Batch 8A

- `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_PAGES_ROOT_ARTIFACT_ALIGNMENT_VALIDATION_20260330.md`

### Batch 8B

- `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_DESIGN_20260330.md`
- `docs/DEDUP_EVAL_REPORTING_PUBLIC_DISCOVERY_SURFACE_ALIGNMENT_VALIDATION_20260330.md`

