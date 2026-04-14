# Training Data Governance Gates Watcher Promotion Plan

日期：2026-04-14

## 背景

`Governance Gates` 已经满足进入 watcher-required 默认集合的前置条件：

- 仓库已切换为 public
- `Governance Gates` 已在 GitHub-hosted `pull_request` run 中真实成功
- 本地 `make validate-training-governance` 与 `make test-training-governance` 持续通过

因此，本批目标从“验证能跑”切换为“把它纳入默认 watcher 视野”。

---

## 总目标

把 `Governance Gates` 从“独立可见 workflow”提升为“watcher-required 默认 workflow”。

具体目标：

1. 更新默认 required workflow 集合
2. 更新 workflow identity / inventory 脚本默认列表
3. 更新显式传参的 workflow inventory 调用点
4. 保持训练治理门禁本身不回退

---

## 非目标

本批不做以下事项：

- 不修复与 `Governance Gates` 无关的历史 workflow identity 失败
- 不把 `Claude Code CLI` 接入 watcher 或 CI 主路径
- 不修改 `Governance Gates` workflow 本身的步骤逻辑

---

## 变更范围

### 1. Makefile

更新：

- `CI_WATCH_REQUIRED_WORKFLOWS`

新增：

- `Governance Gates`

目的：

- 让本地 watcher / guardrail 默认把该 workflow 视为 required

### 2. Workflow identity invariants

更新：

- `scripts/ci/check_workflow_identity_invariants.py`

变更：

- 新增 `governance-gates.yml -> Governance Gates` spec
- 默认 required workflow CSV 增加 `Governance Gates`

目的：

- workflow identity 守护逻辑不再把 `Governance Gates` 视为“额外但未治理”的 workflow

### 3. Workflow inventory report

更新：

- `scripts/ci/generate_workflow_inventory_report.py`

变更：

- 默认 required workflow CSV 增加 `Governance Gates`

目的：

- inventory 报告与 watcher 默认集合一致

### 4. 显式传参调用点

更新：

- `.github/workflows/stress-tests.yml`

变更：

- 显式 `--ci-watch-required-workflows` 参数加入 `Governance Gates`

目的：

- 避免默认列表与显式列表分叉

---

## 验证策略

### 必做

- `make validate-training-governance`
- `python scripts/ci/generate_workflow_inventory_report.py ...`

### 记录但不阻塞

- `python scripts/ci/check_workflow_identity_invariants.py`

说明：

- 该脚本当前存在与 `Governance Gates` 无关的历史 wrapper input 断言失败
- 本批只要求确认：
  - `governance-gates.yml` 自身 identity 为 `ok`
  - 默认 required workflow 集合已包含 `Governance Gates`

---

## 风险与控制

### 风险 1

将 `Governance Gates` 纳入 required 集合后，未来 watcher 会把它当成缺失/失败即报警的 workflow。

控制：

- 由于已经有一轮真实 GitHub-hosted 成功 run，可接受
- 保持 `Governance Gates` workflow 轻量，不引入额外高波动依赖

### 风险 2

workflow identity checker 当前对别的历史 wrapper 仍然失败，容易被误读为 promotion 失败。

控制：

- 在验证文档中明确区分：
  - `Governance Gates` promotion 成功
  - wrapper identity failures 为 pre-existing unrelated issues

---

## 完成标准

满足以下条件即可视为本批完成：

1. 默认 required workflow 列表包含 `Governance Gates`
2. workflow inventory 报告将 `Governance Gates` 识别为 required+ok
3. `make validate-training-governance` 继续通过
4. 形成单独开发/验证文档，记录 residual risk
