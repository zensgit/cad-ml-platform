# DEV_CI_COMMIT_WATCHER_20260221

## 背景与目标
- 在本地通过 `gh` 追踪 CI 时，手工逐条 `gh run watch` 成本高、易漏看。
- 目标是提供一个“按提交 SHA 聚合盯 CI”的脚本，统一等待并输出最终结论。

## 设计说明
- 新增脚本：`scripts/ci/watch_commit_workflows.py`
  - 支持输入：
    - `--sha`（默认 `HEAD`）
    - `--events-csv` / `--event`（默认包含 `push`）
    - `--require-workflows-csv` / `--require-workflow`（必需工作流白名单）
    - `--wait-timeout-seconds`、`--poll-interval-seconds`
    - `--list-limit`、`--print-only`
  - 运行逻辑：
    - 预检 `gh --version` 与 `gh auth status`
    - 拉取 `gh run list`，按 `headSha + event` 过滤
    - 对同名工作流按最大 `databaseId` 去重，仅保留最新 run
    - 持续轮询直至：
      - 所有观察到的工作流 `completed`
      - 且结论均为 `success/skipped`
      - 且满足“必需工作流”集合
  - 运行体验增强：
    - 状态输出改为 `flush=True`，避免长轮询时终端无输出导致“假卡住”。
- Make 集成：
  - `watch-commit-workflows`
  - `validate-watch-commit-workflows`
  - 默认必需工作流集合补齐：`Stress and Observability Checks`。
- README：
  - 增补“按提交 SHA 统一盯 CI”使用说明与回归命令。

## 变更清单
- `scripts/ci/watch_commit_workflows.py`
- `tests/unit/test_watch_commit_workflows.py`
- `tests/unit/test_watch_commit_workflows_make_target.py`
- `Makefile`
- `README.md`

## 验证命令与结果
- `pytest -q tests/unit/test_watch_commit_workflows.py tests/unit/test_watch_commit_workflows_make_target.py`
  - 结果：通过（11 passed）。
- `make -n watch-commit-workflows`
  - 结果：通过，展开命令包含 `scripts/ci/watch_commit_workflows.py` 以及 `--sha/--events-csv/--require-workflows-csv/--wait-timeout-seconds/--poll-interval-seconds/--list-limit` 参数。
- `make watch-commit-workflows CI_WATCH_PRINT_ONLY=1 CI_WATCH_SHA=abc123 CI_WATCH_EVENTS=push,workflow_dispatch CI_WATCH_REQUIRED_WORKFLOWS='CI,Code Quality' CI_WATCH_TIMEOUT=30 CI_WATCH_POLL_INTERVAL=2 CI_WATCH_LIST_LIMIT=50`
  - 结果：通过，仅打印 `gh run list` 预览命令与事件/必需工作流配置，不执行实际轮询。
- `make validate-watch-commit-workflows`
  - 结果：通过（11 passed）。
- `make validate-archive-workflow-dispatcher`
  - 结果：通过（24 passed），确认新改动未回归既有 archive dispatcher 能力。

## 实战验证（2026-02-23）
- 提交：`c2163e4`（`fix: improve commit workflow watcher defaults and streaming output`）
- 命令：
  - `make watch-commit-workflows CI_WATCH_SHA=c2163e4ea91956a5560d73de30ebcadef5b016b9 CI_WATCH_TIMEOUT=5400 CI_WATCH_POLL_INTERVAL=20`
- 结果：脚本实时输出阶段进度并最终返回成功（`all observed workflows completed successfully.`）。
- push 工作流结果（全部 success）：
  - `CI`：`22306644788`
  - `CI Enhanced`：`22306644830`
  - `CI Tiered Tests`：`22306644813`
  - `Code Quality`：`22306644783`
  - `Evaluation Report`：`22306644819`
  - `GHCR Publish`：`22306644789`
  - `Multi-Architecture Docker Build`：`22306644807`
  - `Observability Checks`：`22306644815`
  - `Security Audit`：`22306644809`
  - `Self-Check`：`22306644839`
  - `Stress and Observability Checks`：`22306644811`
