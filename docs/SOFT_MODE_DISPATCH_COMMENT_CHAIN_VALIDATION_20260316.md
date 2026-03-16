# Soft-Mode Dispatch + PR 回写链路开发与验证（2026-03-16）

## 目标

把 `evaluation soft-mode smoke` 的触发与 PR 评论回写串成一条命令链路，减少人工步骤并提升 CI 可观测性。

## 实现

### 1) dispatch 脚本增加可选 PR 回写参数

- 文件：`scripts/ci/dispatch_evaluation_soft_mode_smoke.py`
- 新增参数：
  - `--comment-pr-number`
  - `--comment-pr-auto`
  - `--comment-repo`
  - `--comment-title`
  - `--comment-commit-sha`
  - `--comment-output-json`
  - `--comment-dry-run`
  - `--comment-fail-on-error`
- 行为：
  - 当 `--comment-pr-number > 0` 时，自动调用 `post_soft_mode_smoke_pr_comment.py` 主逻辑。
  - 当设置 `--comment-pr-auto` 且未传 `--comment-pr-number` 时，按 `--ref`（必要时回退到当前 git 分支）自动查询 open PR。
  - 当 `--comment-commit-sha` 为空时，自动尝试注入 `git rev-parse HEAD`。
  - 在输出 JSON 中增加 `pr_comment` 字段，记录回写是否启用、退出码与错误信息。
  - 若设置 `--comment-fail-on-error`，回写失败会将 `overall_exit_code` 置为失败。

### 2) Makefile 接线

- 文件：`Makefile`
- `validate-soft-mode-smoke` 新增对应变量与参数透传：
  - `SOFT_MODE_SMOKE_COMMENT_PR_NUMBER`
  - `SOFT_MODE_SMOKE_COMMENT_PR_AUTO`
  - `SOFT_MODE_SMOKE_COMMENT_REPO`
  - `SOFT_MODE_SMOKE_COMMENT_TITLE`
  - `SOFT_MODE_SMOKE_COMMENT_COMMIT_SHA`
  - `SOFT_MODE_SMOKE_COMMENT_DRY_RUN`
  - `SOFT_MODE_SMOKE_COMMENT_FAIL_ON_ERROR`
  - `SOFT_MODE_SMOKE_COMMENT_OUTPUT_JSON`
- 新增目标：
  - `validate-soft-mode-smoke-auto-pr`（自动开启 `SOFT_MODE_SMOKE_COMMENT_PR_AUTO=1`）

### 3) 单元测试补齐

- 文件：`tests/unit/test_dispatch_evaluation_soft_mode_smoke.py`
  - `test_main_posts_pr_comment_when_requested`
  - `test_main_comment_fail_can_fail_overall`
- 文件：`tests/unit/test_hybrid_calibration_make_targets.py`
  - 断言 `validate-soft-mode-smoke` 包含 comment 相关参数
  - 覆盖 `comment-pr-number / dry-run / fail-on-error` flag 透传

## 验证

### 单测

```bash
pytest -q \
  tests/unit/test_dispatch_evaluation_soft_mode_smoke.py \
  tests/unit/test_post_soft_mode_smoke_pr_comment.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

结果：`42 passed`

### Make 目标展开验证

```bash
make -n validate-soft-mode-smoke \
  SOFT_MODE_SMOKE_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_SMOKE_COMMENT_PR_NUMBER=369 \
  SOFT_MODE_SMOKE_COMMENT_DRY_RUN=1 \
  SOFT_MODE_SMOKE_COMMENT_FAIL_ON_ERROR=1
```

结果：命令行中已包含 `--comment-pr-number / --comment-dry-run / --comment-fail-on-error` 及完整 comment 参数。

### Auto-PR 目标展开验证

```bash
make -n validate-soft-mode-smoke-auto-pr \
  SOFT_MODE_SMOKE_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_SMOKE_COMMENT_DRY_RUN=1
```

结果：包含 `SOFT_MODE_SMOKE_COMMENT_PR_AUTO=1` 透传，并递归调用 `validate-soft-mode-smoke`。

### 真实链路验证（auto PR + dry-run）

```bash
python3 scripts/ci/dispatch_evaluation_soft_mode_smoke.py \
  --repo zensgit/cad-ml-platform \
  --workflow evaluation-report.yml \
  --ref feat/hybrid-blind-drift-autotune-e2e \
  --expected-conclusion success \
  --wait-timeout-seconds 1500 \
  --poll-interval-seconds 3 \
  --list-limit 40 \
  --max-dispatch-attempts 1 \
  --comment-pr-auto \
  --comment-repo zensgit/cad-ml-platform \
  --comment-title "CAD ML Platform - Soft Mode Smoke" \
  --comment-dry-run \
  --output-json reports/experiments/20260316/soft_mode_smoke_auto_pr_dry_run.json
```

结果：

- run_id: `23126562401`
- run_url: `https://github.com/zensgit/cad-ml-platform/actions/runs/23126562401`
- dispatch conclusion: `success`
- `soft_marker_ok=true`
- `pr_comment.enabled=true`
- `pr_comment.pr_number=369`
- `pr_comment.auto_resolve=true`
- `overall_exit_code=0`

## 用法示例

```bash
make validate-soft-mode-smoke \
  SOFT_MODE_SMOKE_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_SMOKE_REF=feat/hybrid-blind-drift-autotune-e2e \
  SOFT_MODE_SMOKE_COMMENT_PR_NUMBER=369 \
  SOFT_MODE_SMOKE_COMMENT_PR_AUTO=1 \
  SOFT_MODE_SMOKE_COMMENT_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_SMOKE_COMMENT_COMMIT_SHA=$(git rev-parse HEAD) \
  SOFT_MODE_SMOKE_COMMENT_DRY_RUN=0 \
  SOFT_MODE_SMOKE_COMMENT_FAIL_ON_ERROR=1 \
  SOFT_MODE_SMOKE_OUTPUT_JSON=reports/experiments/$(date +%Y%m%d)/soft_mode_smoke_with_comment.json
```

## 变更文件

- `scripts/ci/dispatch_evaluation_soft_mode_smoke.py`
- `tests/unit/test_dispatch_evaluation_soft_mode_smoke.py`
- `Makefile`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `docs/SOFT_MODE_DISPATCH_COMMENT_CHAIN_VALIDATION_20260316.md`
