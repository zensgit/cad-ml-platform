# DEV_ARCHIVE_WORKFLOW_DISPATCHER_20260221

## 背景与目标
- 为脚本 `scripts/ci/dispatch_experiment_archive_workflow.py` 提供 Make 集成入口。
- 在 README 的“实验目录归档自动化”章节补充本地 `gh workflow_dispatch` 使用说明。
- 约束 apply 流程必须通过环境变量 `ARCHIVE_APPROVAL_PHRASE` 显式确认。

## 设计说明
- 新增 Make 目标：
  - `archive-workflow-dry-run-gh`：固定 `mode=dry-run`，支持 `ARCHIVE_WORKFLOW_REF`、`ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT`、`ARCHIVE_WORKFLOW_KEEP_DAYS`、`ARCHIVE_WORKFLOW_TODAY`、`ARCHIVE_WORKFLOW_WATCH`、`ARCHIVE_WORKFLOW_PRINT_ONLY` 覆盖。
  - `archive-workflow-apply-gh`：固定 `mode=apply`，支持 `ARCHIVE_WORKFLOW_DIRS_CSV`、`ARCHIVE_WORKFLOW_REQUIRE_EXISTS`、`ARCHIVE_WORKFLOW_WATCH`、`ARCHIVE_WORKFLOW_PRINT_ONLY`；执行前校验 `ARCHIVE_APPROVAL_PHRASE` 非空。
- 传参策略：
  - 通过 CLI 参数传递给脚本（例如 `--mode`、`--ref`），配置值来自 Make 变量（可命令行覆盖）。
  - apply 目标不在仓库中保存审批短语，短语仅在运行时由环境变量注入。

## 使用方式
```bash
# dry-run dispatch
make archive-workflow-dry-run-gh \
  ARCHIVE_WORKFLOW_REF=main \
  ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT=reports/experiments \
  ARCHIVE_WORKFLOW_KEEP_DAYS=7 \
  ARCHIVE_WORKFLOW_TODAY=20260221 \
  ARCHIVE_WORKFLOW_WATCH=1 \
  ARCHIVE_WORKFLOW_PRINT_ONLY=0

# apply dispatch（必须提供审批短语）
ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE \
make archive-workflow-apply-gh \
  ARCHIVE_WORKFLOW_REF=main \
  ARCHIVE_WORKFLOW_EXPERIMENTS_ROOT=reports/experiments \
  ARCHIVE_WORKFLOW_KEEP_DAYS=7 \
  ARCHIVE_WORKFLOW_DIRS_CSV=20260217,20260219 \
  ARCHIVE_WORKFLOW_REQUIRE_EXISTS=true \
  ARCHIVE_WORKFLOW_WATCH=1 \
  ARCHIVE_WORKFLOW_PRINT_ONLY=0
```

## 验证命令与结果
- `make -n archive-workflow-dry-run-gh`
  - 结果：通过，命令中包含 `--mode dry-run` 及共享输入参数。
- `make -n archive-workflow-apply-gh ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE`
  - 结果：通过，命令中包含 `--mode apply`、`--approval-phrase` 及 apply 专属参数。
- `make archive-workflow-dry-run-gh ARCHIVE_WORKFLOW_PRINT_ONLY=1 ARCHIVE_WORKFLOW_WATCH=0`
  - 结果：通过，仅输出 `gh workflow run ...` 命令，不触发实际 dispatch。
- `ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE make archive-workflow-apply-gh ARCHIVE_WORKFLOW_PRINT_ONLY=1 ARCHIVE_WORKFLOW_WATCH=0`
  - 结果：通过，仅输出 apply dispatch 命令，不执行实际触发。
- `pytest -q tests/unit/test_dispatch_experiment_archive_workflow.py tests/unit/test_experiment_archive_workflows.py tests/unit/test_archive_experiment_dirs.py`
  - 结果：通过（12 passed）。

## 备注
- 本次已同时交付：dispatch 脚本、脚本单测、workflow YAML 静态防回归测试、Make/README 集成。
