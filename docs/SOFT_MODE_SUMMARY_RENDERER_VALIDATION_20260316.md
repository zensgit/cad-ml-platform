# Soft-Mode Smoke Summary Renderer 开发与验证（2026-03-16）

## 目标

将 `evaluation-soft-mode-smoke` 的 Step Summary 生成逻辑从 workflow 内嵌 Python 中抽离为可复用脚本，统一本地与 GitHub Actions 的摘要格式。
本轮继续把 soft-mode markdown 调整为先给 `Smoke Verdict` 再展开 attempts/details，并补齐无效 JSON 的 PR comment 回归。
本轮继续补了统一的 `Smoke Snapshot` 段，并为 renderer 本身增加无效 JSON 失败回归。

## 实现

### 1) 新增独立渲染脚本

- 文件：`scripts/ci/render_soft_mode_smoke_summary.py`
- 输入：
  - `--summary-json`
  - `--output-md`（可选）
- 输出：
  - stdout Markdown
  - 可选写入 `.md` 文件
- 覆盖字段：
  - smoke verdict / pr comment status
  - smoke snapshot / failed attempt count / last attempt message
  - overall / dispatch exit code
  - max_dispatch_attempts / retry_sleep_seconds
  - attempts 明细
  - run_id / run_url
  - soft_marker_ok / restore_ok
  - `pr_comment` 状态（若 summary 中存在）

### 2) Make 接线

- 文件：`Makefile`
- 新增变量：
  - `SOFT_MODE_SMOKE_SUMMARY_MD`
- 新增目标：
  - `render-soft-mode-smoke-summary`
  - `validate-render-soft-mode-smoke-summary`
- 更新：
  - `validate-soft-mode-smoke-workflow` 将渲染脚本测试纳入回归

### 3) Workflow 去内联逻辑

- 文件：`.github/workflows/evaluation-soft-mode-smoke.yml`
- 变更：
  - 新增 `Render soft-mode smoke summary markdown` 步骤，产出 `reports/ci/evaluation_soft_mode_smoke_summary.md`
  - `Append summary` 步骤不再内嵌 Python 解析 JSON
  - 改为直接 `cat` 渲染后的 markdown
  - artifact 同时上传 `.json` 与 `.md`

### 4) Workflow 自动解析 PR

- 文件：`.github/workflows/evaluation-soft-mode-smoke.yml`
- 变更：
  - 新增 `Resolve PR number for comment` 步骤
  - 当手工触发未填写 `pr_number` 时，按 `ref` 自动查询 open PR
  - `Comment PR with soft-mode smoke result` 改为消费 `steps.resolve_pr.outputs.pr_number`

### 5) 回归测试

- 新增：`tests/unit/test_render_soft_mode_smoke_summary.py`
- 更新：`tests/unit/test_hybrid_calibration_make_targets.py`
- 更新：`tests/unit/test_evaluation_soft_mode_smoke_workflow.py`
- 更新：`tests/unit/test_comment_soft_mode_smoke_pr_js.py`

## 验证

### 单测

```bash
pytest -q \
  tests/unit/test_render_soft_mode_smoke_summary.py \
  tests/unit/test_evaluation_soft_mode_smoke_workflow.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

结果：`45 passed`

### Make 验证

```bash
make validate-render-soft-mode-smoke-summary
make validate-soft-mode-smoke-workflow
make validate-soft-mode-smoke-comment
```

结果：

- `validate-render-soft-mode-smoke-summary` -> `43 passed`
- `validate-soft-mode-smoke-workflow` -> `56 passed`
- `validate-soft-mode-smoke-comment` -> `3 passed`

### Workflow 自动 PR 解析回归

```bash
pytest -q tests/unit/test_evaluation_soft_mode_smoke_workflow.py
```

结果：`5 passed`

### 真实数据渲染

```bash
python3 scripts/ci/render_soft_mode_smoke_summary.py \
  --summary-json reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.json \
  --output-md reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.md
```

输出文件：

- `reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.md`

关键结果：

- `verdict: ok`
- `overall_exit_code: 0`
- `run_id: 23126562401`
- `pr_comment_pr_number: 369`
- `pr_comment_auto_resolve: True`

## 变更文件

- `scripts/ci/render_soft_mode_smoke_summary.py`
- `tests/unit/test_render_soft_mode_smoke_summary.py`
- `Makefile`
- `.github/workflows/evaluation-soft-mode-smoke.yml`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `tests/unit/test_evaluation_soft_mode_smoke_workflow.py`
- `docs/SOFT_MODE_SUMMARY_RENDERER_VALIDATION_20260316.md`
