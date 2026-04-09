# CI Watcher + Soft-Mode Smoke 融合开发与验证（2026-03-15）

## 目标

1. 执行真实 soft-mode smoke 远端验收（含 PR 上下文）。
2. 将 soft-mode smoke 结果接入 CI watcher 验证报告，形成统一健康视图。

## 远端执行结果

### A. workflow_dispatch 直接触发 `evaluation-soft-mode-smoke.yml`

执行命令：

```bash
gh workflow run evaluation-soft-mode-smoke.yml \
  --ref feat/hybrid-blind-drift-autotune-e2e \
  -f ref=feat/hybrid-blind-drift-autotune-e2e \
  -f expected_conclusion=success \
  -f pr_number=369 \
  -f keep_soft=false \
  -f skip_log_check=false
```

结果：

- `HTTP 404: workflow evaluation-soft-mode-smoke.yml not found on the default branch`
- 结论：当前仓库仍受 GitHub 默认分支可发现性限制，workflow 文件需进入默认分支后才能直接 dispatch。

### B. fallback：直接触发 `evaluation-report.yml` soft smoke

执行命令：

```bash
python3 scripts/ci/dispatch_evaluation_soft_mode_smoke.py \
  --repo zensgit/cad-ml-platform \
  --workflow evaluation-report.yml \
  --ref feat/hybrid-blind-drift-autotune-e2e \
  --expected-conclusion success \
  --wait-timeout-seconds 1500 \
  --poll-interval-seconds 3 \
  --list-limit 40 \
  --max-dispatch-attempts 2 \
  --retry-sleep-seconds 10 \
  --output-json reports/experiments/20260315/soft_mode_smoke_pr369_fallback_dispatch.json
```

结果：

- `run_id=23111551338`
- `run_url=https://github.com/zensgit/cad-ml-platform/actions/runs/23111551338`
- `overall_exit_code=0`
- `soft_marker_ok=True`
- `restore_ok=True`

## 本轮代码增强

### 1) CI watcher 报告接入 soft-smoke 段

- 文件：`scripts/ci/generate_ci_watcher_validation_report.py`
- 新增能力：
  - 新参数：`--soft-smoke-summary-json`
  - 若未显式传入，自动推断 `<summary_dir>/evaluation_soft_mode_smoke_summary.json`
  - 报告新增 `## Soft-Mode Smoke Artifact` 段，输出：
    - overall/dispatch exit code
    - soft marker / restore 状态
    - retry 参数与 attempts 总数
    - run_id / run_url
    - attempt 级细节

### 2) Make 透传支持

- 文件：`Makefile`
- `generate-ci-watch-validation-report` 新增透传：
  - `--soft-smoke-summary-json "$(CI_WATCH_REPORT_SOFT_SMOKE_JSON)"`
- 新变量：
  - `CI_WATCH_REPORT_SOFT_SMOKE_JSON ?=`

### 3) 回归测试更新

- `tests/unit/test_generate_ci_watcher_validation_report.py`
  - 新增/增强 soft-smoke section 断言（存在与缺失两种路径）
- `tests/unit/test_watch_commit_workflows_make_target.py`
  - 新增 make `-n` 断言，确保新参数已接线

## 验证

执行：

```bash
pytest -q tests/unit/test_generate_ci_watcher_validation_report.py tests/unit/test_watch_commit_workflows_make_target.py
python3 -m py_compile scripts/ci/generate_ci_watcher_validation_report.py
make -n generate-ci-watch-validation-report
python3 scripts/ci/generate_ci_watcher_validation_report.py \
  --summary-json reports/ci/watch_commit_06776003e517_summary.json \
  --readiness-json reports/ci/gh_readiness_watch_06776003e517.json \
  --soft-smoke-summary-json reports/experiments/20260315/soft_mode_smoke_pr369_fallback_dispatch.json \
  --output-md reports/experiments/20260315/ci_watcher_with_soft_smoke_20260315.md \
  --date 20260315
```

结果：

- `pytest`: `17 passed`
- `py_compile`: 通过
- `make -n`: 已包含 `--soft-smoke-summary-json ""`
- 生成报告：`reports/experiments/20260315/ci_watcher_with_soft_smoke_20260315.md`
  - 报告中已包含 `Soft-Mode Smoke Artifact` 段及 attempt 级细节。

## 变更文件

- `scripts/ci/generate_ci_watcher_validation_report.py`
- `Makefile`
- `tests/unit/test_generate_ci_watcher_validation_report.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`
- `docs/CI_WATCH_SOFT_SMOKE_FUSION_VALIDATION_20260315.md`
