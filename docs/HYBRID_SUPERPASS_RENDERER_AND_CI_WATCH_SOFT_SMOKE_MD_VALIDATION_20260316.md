# Hybrid Superpass Renderer 与 CI Watcher Soft-Smoke MD 聚合验证（2026-03-16）

## 目标

在不触碰业务模型代码的前提下，继续增强运维与 CI 自动化：

- 为 `hybrid superpass validation` JSON 提供独立 Markdown renderer
- 让 `ci watcher` 报告自动识别并引用 `soft-mode smoke` 的 Markdown artifact

本轮继续补了统一的 `Validation Verdict / Validation Snapshot` 段，并为无效 JSON 输入补齐失败回归。

## 实现

### 1) CI watcher 聚合 soft-smoke markdown artifact

- 文件：`scripts/ci/generate_ci_watcher_validation_report.py`
- 新增参数：
  - `--soft-smoke-summary-md`
- 新增行为：
  - 若未显式传入，则自动推断 `evaluation_soft_mode_smoke_summary.md`
  - 在报告 `Soft-Mode Smoke Artifact` 段落中输出：
    - JSON 路径
    - Markdown 路径 `rendered_markdown=...`
  - stdout 也会打印 `soft_smoke_md=...`

### 2) Make 接线

- 文件：`Makefile`
- 新增变量：
  - `CI_WATCH_REPORT_SOFT_SMOKE_MD`
  - `HYBRID_SUPERPASS_VALIDATION_JSON`
  - `HYBRID_SUPERPASS_VALIDATION_MD`
- 新增目标：
  - `render-hybrid-superpass-validation-summary`
  - `validate-render-hybrid-superpass-validation-summary`
- 更新目标：
  - `generate-ci-watch-validation-report` 增加 `--soft-smoke-summary-md`

### 3) 新增 Hybrid Superpass Validation renderer

- 文件：`scripts/ci/render_hybrid_superpass_validation_summary.py`
- 输入：
  - `--validation-json`
  - `--output-md`
- 输出内容：
  - validation verdict / top errors / top warnings
  - validation snapshot / status / strict / schema_mode / overall_exit_code
  - inputs 路径
  - superpass / gate / calibration 核心 summary 指标
  - errors / warnings 列表

### 4) 回归测试

- 新增：
  - `tests/unit/test_render_hybrid_superpass_validation_summary.py`
- 更新：
  - `tests/unit/test_generate_ci_watcher_validation_report.py`
  - `tests/unit/test_watch_commit_workflows_make_target.py`
  - `tests/unit/test_hybrid_calibration_make_targets.py`

## 验证

### 单测

```bash
pytest -q \
  tests/unit/test_generate_ci_watcher_validation_report.py \
  tests/unit/test_watch_commit_workflows_make_target.py \
  tests/unit/test_render_hybrid_superpass_validation_summary.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

结果：`61 passed`

### Make 验证

```bash
make validate-generate-ci-watch-validation-report
make validate-render-hybrid-superpass-validation-summary
```

结果：

- `validate-generate-ci-watch-validation-report` -> `4 passed`
- `validate-render-hybrid-superpass-validation-summary` -> `43 passed`

### 近真实验证 1：watcher 报告带上 soft-smoke markdown

```bash
python3 scripts/ci/generate_ci_watcher_validation_report.py \
  --summary-json reports/ci/watch_commit_2c2fd8d_summary.json \
  --readiness-json reports/ci/gh_readiness_watch_2c2fd8d.json \
  --soft-smoke-summary-json reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.json \
  --soft-smoke-summary-md reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.md \
  --output-md reports/experiments/20260316/ci_watcher_with_soft_smoke_md_20260316.md \
  --date 20260316
```

结果：

- 输出文件：`reports/experiments/20260316/ci_watcher_with_soft_smoke_md_20260316.md`
- 报告中已包含：
  - `rendered_markdown=reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.md`

### 近真实验证 2：hybrid superpass renderer

通过临时 validation JSON 执行：

```bash
python3 scripts/ci/render_hybrid_superpass_validation_summary.py \
  --validation-json <tmp>/hybrid_superpass_validation.json \
  --output-md reports/experiments/20260316/hybrid_superpass_validation_rendered_20260316.md
```

结果：

- 输出文件：`reports/experiments/20260316/hybrid_superpass_validation_rendered_20260316.md`
- 成功渲染 validation verdict / snapshot / status / summary / warnings

## 变更文件

- `scripts/ci/generate_ci_watcher_validation_report.py`
- `scripts/ci/render_hybrid_superpass_validation_summary.py`
- `Makefile`
- `tests/unit/test_generate_ci_watcher_validation_report.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`
- `tests/unit/test_render_hybrid_superpass_validation_summary.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `docs/HYBRID_SUPERPASS_RENDERER_AND_CI_WATCH_SOFT_SMOKE_MD_VALIDATION_20260316.md`
