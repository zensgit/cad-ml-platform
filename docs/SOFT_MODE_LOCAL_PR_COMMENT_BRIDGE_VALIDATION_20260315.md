# Soft-Mode 本地 PR 回写桥接开发与验证（2026-03-15）

## 目标

在 `evaluation-soft-mode-smoke.yml` 尚未进入默认分支前，提供可在本地直接执行的 PR 评论回写能力，确保 soft-mode smoke 验证结果可闭环到 PR。

## 实现

### 1) 新增本地回写脚本

- 文件：`scripts/ci/post_soft_mode_smoke_pr_comment.py`
- 功能：
  - 输入 `repo` / `pr_number` / `summary_json`
  - 根据标题标记 `CAD ML Platform - Soft Mode Smoke` 自动识别并更新已有 Bot 评论，否则创建新评论
  - 支持 `--dry-run`
  - 支持 `--output-json` 输出结构化执行结果
  - 评论内容包含：
    - overall/dispatch exit code
    - soft marker / restore 状态
    - run_id / run_url
    - attempts 汇总与逐条 attempt 明细

### 2) Make 接线

- 文件：`Makefile`
- 新增变量：
  - `SOFT_MODE_COMMENT_REPO`
  - `SOFT_MODE_COMMENT_PR_NUMBER`
  - `SOFT_MODE_COMMENT_SUMMARY_JSON`
  - `SOFT_MODE_COMMENT_COMMIT_SHA`
  - `SOFT_MODE_COMMENT_TITLE`
  - `SOFT_MODE_COMMENT_DRY_RUN`
  - `SOFT_MODE_COMMENT_OUTPUT_JSON`
- 新增目标：
  - `soft-mode-smoke-comment-pr`
  - `validate-soft-mode-smoke-comment-pr`

### 3) 回归测试

- 新增：`tests/unit/test_post_soft_mode_smoke_pr_comment.py`
  - 覆盖创建评论 / 更新评论 / dry-run 不写入
- 更新：`tests/unit/test_hybrid_calibration_make_targets.py`
  - 增加 Make 参数透传与 validate 目标断言

## 验证

### 命令

```bash
pytest -q tests/unit/test_post_soft_mode_smoke_pr_comment.py tests/unit/test_hybrid_calibration_make_targets.py
make validate-soft-mode-smoke-comment-pr
```

### 结果

- `pytest`：`33 passed`
- `make validate-soft-mode-smoke-comment-pr`：`33 passed`

## 真实执行（PR #369）

### dry-run

```bash
make soft-mode-smoke-comment-pr \
  SOFT_MODE_COMMENT_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_COMMENT_PR_NUMBER=369 \
  SOFT_MODE_COMMENT_SUMMARY_JSON=reports/experiments/20260315/soft_mode_smoke_pr369_fallback_dispatch.json \
  SOFT_MODE_COMMENT_COMMIT_SHA=$(git rev-parse HEAD) \
  SOFT_MODE_COMMENT_DRY_RUN=1 \
  SOFT_MODE_COMMENT_OUTPUT_JSON=reports/experiments/20260315/soft_mode_smoke_pr_comment_dry_run_20260315.json
```

结果：`action=dry_run_create_comment`

### 正式回写

```bash
make soft-mode-smoke-comment-pr \
  SOFT_MODE_COMMENT_REPO=zensgit/cad-ml-platform \
  SOFT_MODE_COMMENT_PR_NUMBER=369 \
  SOFT_MODE_COMMENT_SUMMARY_JSON=reports/experiments/20260315/soft_mode_smoke_pr369_fallback_dispatch.json \
  SOFT_MODE_COMMENT_COMMIT_SHA=$(git rev-parse HEAD) \
  SOFT_MODE_COMMENT_DRY_RUN=0 \
  SOFT_MODE_COMMENT_OUTPUT_JSON=reports/experiments/20260315/soft_mode_smoke_pr_comment_apply_20260315.json
```

结果：

- `action=create_comment`
- `created_comment_id=4063040975`

已通过 `gh api repos/zensgit/cad-ml-platform/issues/369/comments` 读取并确认评论内容落地。

## 变更文件

- `scripts/ci/post_soft_mode_smoke_pr_comment.py`
- `Makefile`
- `tests/unit/test_post_soft_mode_smoke_pr_comment.py`
- `tests/unit/test_hybrid_calibration_make_targets.py`
- `docs/SOFT_MODE_LOCAL_PR_COMMENT_BRIDGE_VALIDATION_20260315.md`
