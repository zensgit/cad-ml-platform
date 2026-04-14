# Training Data Governance Gates GitHub Rollout Verification

**日期**: 2026-04-14  
**验证范围**: 远端提交路径、PR 创建、`Governance Gates` 首轮 GitHub-hosted 触发、是否具备 watcher promotion 前提

---

## 1. 提交路径验证

### 1.1 直推 main

执行：

```bash
git push origin main
```

结果：

- 失败

远端返回：

```text
remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: - Changes must be made through a pull request.
```

结论：

- 远端 `main` 为 protected branch
- 正确提交路径必须是分支 + PR

### 1.2 分支推送

执行：

```bash
git switch -c submit/local-main-20260414
git push -u origin submit/local-main-20260414
```

结果：

- 成功

远端分支：

- `origin/submit/local-main-20260414`

---

## 2. PR 创建验证

执行结果：

- PR 成功创建

PR 信息：

- PR: `#398`
- 标题: `feat: submit local training governance and model stack`
- URL: `https://github.com/zensgit/cad-ml-platform/pull/398`
- base: `main`
- head: `submit/local-main-20260414`
- state: `OPEN`
- draft: `false`

结论：

- 当前本地 `HEAD` 已经进入标准审核路径

---

## 3. Governance Gates 首轮真实运行验证

### 3.1 触发结果

通过：

```bash
gh run list --workflow "Governance Gates" --branch submit/local-main-20260414
```

观测到真实 run：

- workflow: `Governance Gates`
- event: `pull_request`
- run id: `24388372839`
- URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/24388372839`
- status: `completed`
- conclusion: `failure`

### 3.2 失败定位

通过：

```bash
gh run view 24388372839
```

得到关键信息：

```text
ANNOTATIONS
X The job was not started because recent account payments have failed or your spending limit needs to be increased.
```

对应 job：

- job name: `training-governance`
- job id: `71227697756`

关键判断：

- 失败发生在 job 启动前
- 不是 workflow YAML parse 错误
- 不是 `make validate-training-governance` 失败
- 不是 `make test-training-governance` 失败
- 是 GitHub Actions 账户账单 / spending limit 问题

### 3.3 本地与远端结论对照

本地：

- `make validate-training-governance` 通过
- `make test-training-governance` 通过，`80 passed`

远端：

- `Governance Gates` 被真实触发
- 但 runner 未真正开始执行

结论：

- 当前没有证据表明 `Governance Gates` 的代码路径有问题
- 当前也没有证据支持“GitHub-hosted 首轮成功”

---

## 4. Watcher Promotion 判定

### 当前结论

**暂不提升到 watcher-required。**

原因：

1. 还没有至少一轮成功的 GitHub-hosted `Governance Gates`
2. 当前失败来自平台账单/额度，不是 workflow 逻辑
3. 若现在提升到 watcher-required，会把平台条件问题引入 watcher 误报

### 若后续要提升

需要先满足：

- GitHub Actions 账单/额度问题解决
- `Governance Gates` 至少成功运行一轮
- 再修改：
  - `Makefile` 的 `CI_WATCH_REQUIRED_WORKFLOWS`
  - watcher / identity / inventory 相关 guardrail

---

## 5. Claude Code CLI 说明

本轮没有把 `Claude Code CLI` 放入提交流程或远端验证主路径。

原因：

- 提交路径和 GitHub run 观察可以完全由 `git` + `gh` 完成
- 当前核心阻塞是 GitHub Actions 账单/额度，不是本地开发工具能力

结论：

- `Claude Code CLI` 仍然只适合 sidecar 审阅
- 不适合介入此次 GitHub-hosted rollout 判定主链

---

## 6. 最终结论

| 项目 | 状态 |
|------|------|
| 直推 main | ✗ 被 protected branch 拦截 |
| 分支推送 | ✓ |
| PR 创建 | ✓ |
| `Governance Gates` 首轮真实触发 | ✓ |
| `Governance Gates` 首轮真实成功 | ✗ |
| 失败归因已明确 | ✓ GitHub Actions billing/spending limit |
| watcher promotion 条件具备 | ✗ |

**结论**：

这轮 GitHub rollout 已经完成了“首次真实触发”验证，但还没有完成“首次真实成功”验证。  
下一步正确动作不是改 workflow 代码，而是先处理 GitHub Actions 账户账单/额度问题，然后重新触发 `Governance Gates`。
