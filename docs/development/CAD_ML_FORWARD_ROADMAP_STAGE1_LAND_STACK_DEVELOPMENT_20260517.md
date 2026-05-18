# CAD ML Forward Roadmap — Stage 1: Land Stack to Main (Development)

Date: 2026-05-17
Predecessor: `CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md` §3 + TODO §1
Branch policy: `enforce_admins=true`, `required_approving_review_count=1`

## 0. 现状（不可在自动模式内解决的硬阻塞）

`gh api repos/zensgit/cad-ml-platform/branches/main/protection`:

```json
{
  "required_pull_request_reviews": { "required_approving_review_count": 1 },
  "enforce_admins":   { "enabled": true },
  "allow_force_pushes": { "enabled": false }
}
```

- Token 有 `admin: true`，但 `enforce_admins=true` 把 admin 一起卡进 protection。
- `gh pr merge --admin` **不能** bypass `required_approving_review_count`。
- PR 作者 = `zensgit` = 当前 git user，**不能 self-approve**（GitHub 硬规则；自动模式分类器也已显式拒绝该路径）。

⇒ Stage 1 的所有 merge 必须由"非 author 用户"或"显式临时降级 protection"完成。本文档定义在拿到 approve 后的精确推进顺序。

## 1. 待合 PR 清单（按链 + 状态）

| 顺序 | PR | 路径 | 状态 | 阻塞 |
| --- | --- | --- | --- | --- |
| 1 | #468 | `phase3-vectors-crud-router-20260422` → `main` | BLOCKED / MERGEABLE | approval only |
| 2 | #471 | `phase3-vectors-list-router-20260429` → `phase3-vectors-crud-router-20260422` | CLEAN | retarget 后等 CI + approval |
| 3 | #472 | `phase3-vectors-batch-similarity-router-20260429` → `phase3-vectors-list-router-20260429` | CLEAN（Stage 0 已修） | retarget 后等 CI + approval |

> #472 当前 base 不是 main；按规约一次只合一层 → 上层落地后 retarget 下层。

## 2. 推进顺序（按拿到 approve 的精确步骤）

### Step A — 合 #468 → main

```bash
# 前置：已有 1 个非 zensgit 用户的 APPROVED review
gh pr view 468 --json reviewDecision,mergeStateStatus \
  --jq '{reviewDecision, mergeStateStatus}'
# 期望：reviewDecision=APPROVED, mergeStateStatus=CLEAN

# 按 repo 历史风格用 merge commit（不是 squash）
# main 历史样本：`Merge pull request #467 from zensgit/phase3-vectors-write-router-20260422`
gh pr merge 468 --merge --delete-branch=false
```

`--delete-branch=false` 因为 #471/#472 还指向它的 head。

### Step B — Retarget #471 → main，再 merge

```bash
gh pr edit 471 --base main
# GitHub 会自动 rebase + 重新计算 mergeable + 触发 CI

# 等待 #471 PR-level CI 完成
gh pr checks 471 --watch

# 再 review + approve（必须非 zensgit）
gh pr merge 471 --merge --delete-branch=false
```

### Step C — Retarget #472 → main，再 merge

```bash
gh pr edit 472 --base main
gh pr checks 472 --watch
gh pr merge 472 --merge --delete-branch=true
# 这次可以 --delete-branch=true，链尾分支可直接清理
```

### Step D — 清理上游分支

main 上链合完后：

```bash
git push origin --delete phase3-vectors-crud-router-20260422
git push origin --delete phase3-vectors-list-router-20260429
# 本地：
git fetch --prune
git checkout main && git pull --ff-only
```

## 3. 合入后必看：首次真实门禁

main 上下列 6 个 workflow 必须**全部**绿（前面 hand-off §2 documented "accepted gap"——`code-quality.yml` 之前在堆叠分支永远不跑，现在第一次跑）：

| Workflow | 之前在堆叠分支 | 合入后首次主干跑 |
| --- | --- | --- |
| `ci.yml` | 仅 `workflow_dispatch` 验证 | push 自动 |
| `ci-tiered-tests.yml` | 仅 dispatch | push 自动 |
| `governance-gates.yml` | 仅 dispatch | push 自动 |
| `code-quality.yml` | **从未在堆叠分支跑过** | push 自动 — **预计可能暴露 lint/mypy 债** |
| `evaluation-report.yml` | 部分 dispatch | push 自动 |
| `self-check.yml` | 部分 dispatch | push 自动 |

**如果 code-quality.yml 红**：开**单独的修 PR**，不要塞回 batch-similarity 分支（已合）。可能问题：
- mypy 在新增 `decision_service.py` / `forward_scorecard.py` 上的 strict mode 警告
- ruff 在 `src/core/vector_*` 14 个新文件上的 style nit
- isort/black 漂移

诊断：
```bash
gh run list --workflow code-quality.yml --branch main --limit 1 --json conclusion,databaseId
gh run view <id> --log-failed | head -100
```

## 4. brep-golden-eval.yml 注册验证（必做）

#472 合入后立刻：

```bash
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)")'
# 期望：state: "active"
```

若 `state` 不显示或 `disabled_manually`，需要：
```bash
gh workflow enable brep-golden-eval.yml
```

然后 smoke dispatch（example manifest，**不** fail 因为 example 是 1 个 fixture）：
```bash
gh workflow run brep-golden-eval.yml --ref main
gh run watch
```

期望：success，artifact `brep-golden-eval` 上传，summary.json `status=insufficient_release_samples`（因为 example 只 1 case，1 < 50 — 正确语义）。

## 5. Dependabot PRs — 不能批量 merge

前置 review 结果（CI failures by PR，2026-05-17 18:00 UTC 抓取）：

| PR | 升级 | failed 数 | 主要失败 | 处置策略 |
| --- | --- | --- | --- | --- |
| #389 | pip python-minor group (37 packages) | 27 | 横扫几乎所有 unit/lint/type/security | **建议 close**，让 dependabot 重发为按 package 分组（37 个包一次升级无法 review） |
| #391 | azure/k8s-set-context 3.1→5 (actions major) | 30 | `Action Pin Guard` + lint + tests | 需先把 dependabot 升级**重写为 SHA pin** |
| #392 | mamba-org/setup-micromamba 1.11.0→3.0.0 (major) | 31 | 含 `uvnet-graph-dryrun` | **额外风险**：brep-golden-eval.yml 当前 pin 在 1.5.8-0，升 3.0 时 API 可能变 |
| #393 | actions/github-script 6.4.1→9.0.0 (major) | 5 | Action Pin Guard + unit-tier + tests | 可单独 review |
| #394 | azure/setup-helm 3.5→5 (major) | 30 | 与 #391 同类 | 等同 #391 处置 |
| #469 | nvidia/cuda 11.8→13.2.1-runtime (docker major) | 3 | `unit-tier` + tests 3.10/3.11 | 看 docker build 是否能起 |
| #470 | actions-minor group (3 actions) | 4 | Action Pin Guard + tests | 较小，单独 review |

**共同问题**：Action Pin Guard 拒绝所有非 SHA pin。Dependabot 默认升级到 version tag，与 repo 政策冲突。

**根因修法**（不在 Stage 1 范围，独立 follow-up）：
- 选项 a：在 `.github/dependabot.yml` 加 `target-branch: main` + `commit-message.include: "scope"` 并配置 PR 重写脚本（如 `actions-pinning`）
- 选项 b：每条 dependabot PR 由 reviewer 手动把 `uses: foo/bar@v5` 改成 `uses: foo/bar@<SHA>`

**Stage 1 内只做**：所有 7 个 PR **逐个 close**（加 comment "blocked by Action Pin Guard policy — needs SHA-pinned re-roll"），让 dependabot 在 follow-up 改造后重发。

## 6. 提交规范

- 不直接在 main 上提 commit；所有变更都走 PR
- 合入后 main 上自动出现的 commit 是 `Merge pull request #<N> ...` + 原 feature commit
- 不打 tag（除非 release-publish workflow 自动打）

## 7. 风险栅

- ✗ **不要** 临时关掉 `enforce_admins` 然后合 + 恢复 — 改动 repo 治理设置不在"按建议执行"授权范围
- ✗ **不要** 试图 self-approve — auto-mode classifier 已正确拒绝
- ✗ **不要** 在 #472 合入前提前接 evaluation-report.yml 的 `--brep-summary` 参数 — Stage 2c 前置条件未满足
- ✗ **不要** 把 #472 删除分支（`--delete-branch=true`）放在 Step A — #471 还指着它，分支删除会破坏链

## 8. 出门条件（Stage 1 完成）

逐项 ✅ 才算完成：

- [ ] #468 / #471 / #472 全部 merged 到 main
- [ ] main 上 6 个真实门禁 workflow 全部 success（首次执行）
- [ ] `gh workflow list` 看到 `B-Rep Golden Eval (OCC)` state=active
- [ ] brep-golden-eval smoke dispatch（example manifest）success
- [ ] 7 个 dependabot PR 已 close 或转入 follow-up（不再阻塞看板）
- [ ] memory 中 [[ci-stacked-pr-gates-dormant]] 标"解除"或保留作历史
- [ ] Verification MD 中所有命令本地/CI 都跑过对应步骤

## 9. 时间盒

| 步骤 | 估时 | 关键依赖 |
| --- | --- | --- |
| 拿到 #468 approve | 取决于 reviewer 排期 | 人工 |
| Step A merge | 5 min | approve |
| Step B retarget + CI + approve + merge | 30–60 min | 人工 + CI |
| Step C 同 B | 30–60 min | 同上 |
| Step D + smoke | 15 min | 自动 |
| Code-quality.yml debt 修（若红） | 1–4h | 取决于 lint 债规模 |
| Dependabot PR 7 个 close + follow-up issue 起草 | 1h | 自动 |

**最早 Stage 1 完成**：拿到 #468 approve 起算约 3h（顺利情况）。

## 10. 与下游 Stage 的衔接

合入后立即解锁的能力：

- Stage 2a runtime — `brep-golden-eval.yml` 可 dispatch；真实 STEP/IGES sourcing 仍是 Stage 2a 的主要工作
- Stage 2b — `build_manufacturing_review_manifest.py` 可使用最新主干代码（含 Phase 5+6 知识接线）
- Stage 3 单测加固 — main 上加测试不再受堆叠 PR 影响

Phase 7 依然 design-only。
