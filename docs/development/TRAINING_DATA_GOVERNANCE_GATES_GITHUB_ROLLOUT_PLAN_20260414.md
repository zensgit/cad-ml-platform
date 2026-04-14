# Training Data Governance Gates GitHub Rollout Plan

日期：2026-04-14

## 背景

本地 `Training Data Governance Gates` 已经完成：

- 独立 workflow
- 本地 `make validate-training-governance`
- 本地 `make test-training-governance`
- `.venv311` 环境收口

但是否将 `Governance Gates` 提升为 watcher-required workflow，仍然需要一个真实 GitHub-hosted 运行结果作为前置依据。

---

## 目标

完成 `Governance Gates` 的首次真实 GitHub rollout 验证，并基于真实结果决定是否进入下一阶段：

1. 通过 PR 提交远端
2. 观察 workflow 是否被真实触发
3. 判断失败是“代码问题”还是“平台/账户问题”
4. 只有在真实 run 成功后，才考虑提升到 watcher-required 集合

---

## 非目标

本批不做以下事项：

- 不把 `Governance Gates` 立即加入 `CI_WATCH_REQUIRED_WORKFLOWS`
- 不修改 watcher identity/inventory 相关 guardrail
- 不把 GitHub Actions 账户账单问题误判为 workflow 代码问题

---

## 实施步骤

### Phase 1 - Protected Main 提交路径

由于远端 `main` 是 protected branch，不能直接：

```bash
git push origin main
```

因此必须改走：

1. 从当前 `HEAD` 切提交分支
2. 推送远端分支
3. 创建 PR

### Phase 2 - 首轮真实运行观察

观察对象：

- PR 是否创建成功
- `Governance Gates` 是否被 pull_request 事件触发
- run 是否真正进入 job 执行
- 若失败，失败发生在：
  - workflow parse / dispatch
  - runner startup
  - dependency install
  - gate/test logic
  - GitHub 平台/账单限制

### Phase 3 - 升级判定

只有满足以下条件才考虑升级到 watcher-required：

1. 至少一轮 GitHub-hosted `Governance Gates` 成功
2. 失败时能提供稳定 artifact / summary
3. 运行时间和失败模式可接受
4. 不会引入新的 watcher identity/inventory 冲突

---

## 预期触点

### GitHub

- PR 页面
- Actions run 页面
- `gh pr view`
- `gh run list`
- `gh run view`

### 后续若要升级 watcher-required

需要二次变更的触点将包括：

- `Makefile` 中的 `CI_WATCH_REQUIRED_WORKFLOWS`
- `scripts/ci/watch_commit_workflows.py`
- `scripts/ci/check_workflow_identity_invariants.py`
- `scripts/ci/generate_workflow_inventory_report.py`
- 相关 workflow identity/guardrail 文档

但这些改动在首次真实成功 run 之前不做。

---

## 风险与控制

### 风险 1

远端 `main` 保护规则阻止直推。

控制：

- 统一走 PR 路径

### 风险 2

首次失败来自 GitHub 平台条件，而不是代码本身。

控制：

- 明确区分 “workflow logic failure” 与 “platform/billing failure”
- 不基于错误归因做 watcher promotion

### 风险 3

当前本地分支超前远端不止 2 个提交。

控制：

- PR 正文显式列出本地超前提交
- 用 PR 而不是直推，交给现有 CI 统一验证

---

## 完成标准

满足以下条件即可视为本批 GitHub rollout 完成：

1. 成功创建远端分支
2. 成功创建 PR
3. 成功观察到 `Governance Gates` 首轮真实 run
4. 能给出明确结论：
   - 是否是 workflow 代码问题
   - 是否可继续推进 watcher-required

---

## 当前决策原则

在没有至少一轮成功的 GitHub-hosted `Governance Gates` 之前：

- 保持它为独立可见 workflow
- 不升级到 watcher-required
- 优先解决平台侧阻塞，再评估流程级升级
