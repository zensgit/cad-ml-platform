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

## v2 稳健性增强
- 新增 Make 默认变量：`ARCHIVE_WORKFLOW_WAIT_TIMEOUT ?= 120`、`ARCHIVE_WORKFLOW_POLL_INTERVAL ?= 3`。
- 两个 dispatch 目标统一透传：
  - `--wait-timeout-seconds "$(ARCHIVE_WORKFLOW_WAIT_TIMEOUT)"`
  - `--poll-interval-seconds "$(ARCHIVE_WORKFLOW_POLL_INTERVAL)"`
- 语义约束：两个参数主要在 `ARCHIVE_WORKFLOW_WATCH=1` 时生效，用于控制最长等待时长与轮询频率。
- 调度脚本增强：
  - dispatch 前执行 `gh --version` + `gh auth status` 预检。
  - watch 模式改为“已知 run id 集合 + 新 run id 发现”机制，降低并发触发下误跟踪旧 run 的概率。
  - 超时与轮询间隔可配置（默认 `120s` / `3s`）。
- 新增一键回归门：`make validate-archive-workflow-dispatcher`。
  - 覆盖 dispatcher 单测 + workflow YAML 安全门测试 + Make 目标透传测试。

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
- `make -n archive-workflow-dry-run-gh ARCHIVE_WORKFLOW_WATCH=1`
  - 结果：通过，展开命令包含 `--wait-timeout-seconds "120"` 与 `--poll-interval-seconds "3"`。
- `make -n archive-workflow-apply-gh ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE ARCHIVE_WORKFLOW_WATCH=1`
  - 结果：通过，展开命令包含 `--wait-timeout-seconds "120"` 与 `--poll-interval-seconds "3"`，并保留 `--approval-phrase`。
- `make -n archive-workflow-dry-run-gh`
  - 结果：通过，命令中包含 `--mode dry-run` 及共享输入参数。
- `make -n archive-workflow-apply-gh ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE`
  - 结果：通过，命令中包含 `--mode apply`、`--approval-phrase` 及 apply 专属参数。
- `make archive-workflow-dry-run-gh ARCHIVE_WORKFLOW_PRINT_ONLY=1 ARCHIVE_WORKFLOW_WATCH=1 ARCHIVE_WORKFLOW_WAIT_TIMEOUT=30 ARCHIVE_WORKFLOW_POLL_INTERVAL=2`
  - 结果：通过，打印 `gh workflow run ...` + `gh run list ...` + `gh run watch <run_id> --exit-status`，不触发实际 dispatch。
- `ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE make archive-workflow-apply-gh ARCHIVE_WORKFLOW_PRINT_ONLY=1 ARCHIVE_WORKFLOW_WATCH=1 ARCHIVE_WORKFLOW_WAIT_TIMEOUT=30 ARCHIVE_WORKFLOW_POLL_INTERVAL=2`
  - 结果：通过，打印 apply dispatch/list/watch 命令，不执行实际触发。
- `pytest -q tests/unit/test_dispatch_experiment_archive_workflow.py tests/unit/test_experiment_archive_workflows.py tests/unit/test_archive_experiment_dirs.py`
  - 结果：通过（20 passed）。
- `make validate-archive-workflow-dispatcher`
  - 结果：通过（24 passed）。

## CI 收口（2026-02-21）
- 提交：`75b27fd`（`test: add make-level regression gate for archive workflow dispatcher`）
- 结论：该提交对应的 push 工作流全部成功。
- 关键工作流：
  - `CI`：run `22257919241`（success）
  - `CI Enhanced`：run `22257919249`（success）
  - `CI Tiered Tests`：run `22257919245`（success）
  - `Code Quality`：run `22257919247`（success）
  - `Multi-Architecture Docker Build`：run `22257919244`（success）
  - `Security Audit`：run `22257919248`（success）
  - `Observability Checks`：run `22257919259`（success）
  - `Stress and Observability Checks`：run `22257919253`（success）
  - `Self-Check`：run `22257919242`（success）
  - `GHCR Publish`：run `22257919243`（success）
  - `Evaluation Report`：run `22257919240`（success）

## 备注
- 本次已同时交付：dispatch 脚本、脚本单测、workflow YAML 静态防回归测试、Make/README 集成。
