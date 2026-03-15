# Soft-Mode Smoke 并行开发与验证（2026-03-15）

## 本轮目标

1. 为 `evaluation-soft-mode-smoke` 增加 PR 结果自动回写能力（PR-only 场景）。
2. 完成 `dispatch_evaluation_soft_mode_smoke.py` 的重试机制收口，并补齐单测。

## 设计与实现

### A. PR 自动回写（Soft-Mode Smoke）

- 新增脚本：`scripts/ci/comment_soft_mode_smoke_pr.js`
  - 导出函数：`commentSoftModeSmokePR({ github, context, process })`
  - 读取环境变量：
    - `SOFT_SMOKE_SUMMARY_JSON`
    - `SOFT_SMOKE_TRIGGER_PR`
  - 容错策略：
    - summary 文件不存在 -> `console.warn` 并跳过
    - summary JSON 非法 -> `console.warn` 并跳过
    - 无有效 PR number -> `console.warn` 并跳过
  - 评论策略：
    - 标题关键字：`CAD ML Platform - Soft Mode Smoke`
    - 若已有 bot 同标题评论则更新，否则新建
  - 评论内容包含：
    - `overall_exit_code`
    - `soft_marker_ok`
    - `restore_ok`
    - `run_id` / `run_url`
    - attempts 总数与每次 `dispatch_exit_code` / `soft_marker_ok`

- 更新 workflow：`.github/workflows/evaluation-soft-mode-smoke.yml`
  - `workflow_dispatch.inputs` 新增 `pr_number`
  - `soft-mode-smoke` job 增加权限：
    - `issues: write`
    - `pull-requests: write`
  - 新增步骤 `Comment PR with soft-mode smoke result`
    - 条件：`workflow_dispatch` 且 `pr_number != ''`
    - action pin：`actions/github-script@d7906e4ad0b1822421a7e6a35d5ca353c962f410`
    - 调用：
      - `SOFT_SMOKE_SUMMARY_JSON=reports/ci/evaluation_soft_mode_smoke_summary.json`
      - `SOFT_SMOKE_TRIGGER_PR=${{ github.event.inputs.pr_number }}`
      - `commentSoftModeSmokePR(...)`

### B. Dispatch 重试机制

- 文件：`scripts/ci/dispatch_evaluation_soft_mode_smoke.py`
- 已支持参数：
  - `--max-dispatch-attempts`（默认 `1`）
  - `--retry-sleep-seconds`（默认 `15`）
- 行为：
  - dispatch + marker 检查失败时自动重试
  - 记录每次 attempt 详情到 `attempts[]`
  - 成功即提前退出
  - 全部失败时保留最后一次 attempt 的 dispatch 信息
  - 变量恢复逻辑保持不变（`finally` 中执行）

## 测试与验证

### 执行命令

```bash
node --check scripts/ci/comment_soft_mode_smoke_pr.js
pytest -q tests/unit/test_dispatch_evaluation_soft_mode_smoke.py tests/unit/test_evaluation_soft_mode_smoke_workflow.py tests/unit/test_hybrid_calibration_make_targets.py
make validate-soft-mode-smoke-workflow
```

### 结果

- `node --check`：通过
- `pytest`：`34 passed`（含 warning 1 条，来自第三方包 `starlette/formparsers.py` 的 PendingDeprecationWarning）
- `make validate-soft-mode-smoke-workflow`：通过（同批用例 `34 passed`）

## 变更文件

- `scripts/ci/comment_soft_mode_smoke_pr.js`（新增）
- `.github/workflows/evaluation-soft-mode-smoke.yml`
- `scripts/ci/dispatch_evaluation_soft_mode_smoke.py`
- `tests/unit/test_dispatch_evaluation_soft_mode_smoke.py`
- `tests/unit/test_evaluation_soft_mode_smoke_workflow.py`
- `docs/SOFT_MODE_SMOKE_PARALLEL_DEV_VALIDATION_20260315.md`（新增）

## 说明

- 本轮只收口 soft-mode smoke 相关链路，未触碰其他功能分支内容。
