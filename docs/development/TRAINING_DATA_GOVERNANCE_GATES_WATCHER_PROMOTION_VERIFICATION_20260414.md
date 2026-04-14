# Training Data Governance Gates Watcher Promotion Verification

**日期**: 2026-04-14  
**验证范围**: `Governance Gates` 首轮 GitHub-hosted 成功、仓库公开化后的可运行性、watcher-required 默认集合提升

---

## 1. 前置事实

### 1.1 仓库可见性

验证结果：

- `visibility = PUBLIC`
- `isPrivate = false`

结论：

- 之前 GitHub Actions job 启动前被账单/额度拦截的问题，在公开仓库后已解除

### 1.2 首轮 GitHub-hosted 成功运行

成功 run：

- workflow: `Governance Gates`
- run id: `24388472620`
- event: `pull_request`
- branch: `submit/local-main-20260414`
- URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/24388472620`
- conclusion: `success`

job 结果：

- job: `training-governance`
- job id: `71228857537`
- duration: `1m6s`

步骤结果：

- `Install dependencies` ✓
- `Run governance invariant checks` ✓
- `Run governance regression tests` ✓
- `Upload governance logs` ✓
- `Append governance summary` ✓

结论：

- `Governance Gates` 已具备真实 GitHub-hosted 成功运行证据

---

## 2. Watcher Promotion 变更

### 2.1 Makefile 默认 required 集合

更新：

- `CI_WATCH_REQUIRED_WORKFLOWS`

结果：

- 默认 required workflow 列表现在包含 `Governance Gates`

### 2.2 Workflow identity invariants

更新：

- `scripts/ci/check_workflow_identity_invariants.py`

结果：

- 新增 `governance-gates.yml -> Governance Gates`
- `require_ci_watch=True`
- 默认 required workflow CSV 包含 `Governance Gates`

### 2.3 Workflow inventory report

更新：

- `scripts/ci/generate_workflow_inventory_report.py`

结果：

- 默认 required workflow CSV 包含 `Governance Gates`

### 2.4 显式调用点

更新：

- `.github/workflows/stress-tests.yml`

结果：

- 显式 `--ci-watch-required-workflows` 也包含 `Governance Gates`

---

## 3. 本地验证

### 3.1 Training governance 门禁

执行：

```bash
make validate-training-governance
```

结果：

```json
{"status": "ok", "checks_count": 10, "violations_count": 0}
```

结论：

- 训练数据治理主门禁未受 watcher promotion 影响

### 3.2 Workflow inventory 报告

执行：

```bash
.venv311/bin/python scripts/ci/generate_workflow_inventory_report.py \
  --output-json /tmp/workflow_inventory_report.json \
  --output-md /tmp/workflow_inventory_report.md
```

结果摘要：

- `workflow_count = 36`
- `required_count = 11`
- `missing_required_count = 0`
- `non_unique_required_count = 0`
- `Governance Gates: status=ok files=governance-gates.yml`

结论：

- `Governance Gates` 已被 inventory 系统视为 required 且映射正确

### 3.3 Workflow identity invariants

执行：

```bash
.venv311/bin/python scripts/ci/check_workflow_identity_invariants.py
```

结果：

- `governance-gates.yml - ok`
- `CI_WATCH_REQUIRED_WORKFLOWS - ok`
- 但整体命令仍失败

失败原因：

- `evaluation-report.yml` 缺少若干 `workflow_dispatch` inputs 断言
- `hybrid-superpass-e2e.yml` 缺少若干 `workflow_dispatch` inputs 断言

关键判断：

- 这些失败是 **pre-existing unrelated issues**
- 不由 `Governance Gates` promotion 引入
- 不影响 `Governance Gates` 自身被 required+ok 识别

---

## 4. Claude Code CLI 说明

本轮没有把 `Claude Code CLI` 用于 watcher promotion 主链验证。

原因：

- 本轮需要的是本地脚本和 GitHub workflow 真实运行结果
- 这些验证不依赖 Claude CLI
- 当前沙箱网络受限，Claude CLI 不适合作为主路径

结论：

- 可以调用，但仍建议只作为 sidecar 审阅工具

---

## 5. 最终结论

| 项目 | 状态 |
|------|------|
| 仓库公开化后 `Governance Gates` 可真实运行 | ✓ |
| 首轮 GitHub-hosted 成功 run | ✓ |
| Makefile 默认 watcher-required 集合包含 `Governance Gates` | ✓ |
| inventory 默认 required 集合包含 `Governance Gates` | ✓ |
| stress-tests 显式 required 集合包含 `Governance Gates` | ✓ |
| training governance 本地门禁继续通过 | ✓ |
| identity checker 全绿 | ✗（存在历史 unrelated failures） |

**结论**：

`Governance Gates` 已完成 watcher-required 默认集合提升。  
当前唯一残留问题不是本批引入的，而是历史 `workflow identity invariants` 对 `evaluation-report` / `hybrid-superpass-e2e` 的旧断言失败，需要后续单独处理。
