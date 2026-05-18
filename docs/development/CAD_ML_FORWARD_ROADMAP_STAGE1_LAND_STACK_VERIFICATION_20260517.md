# CAD ML Forward Roadmap — Stage 1: Land Stack to Main (Verification)

Date: 2026-05-17
Companion: `CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_DEVELOPMENT_20260517.md`

**Principle**: Stage 1 是"治理 + 合入"动作，验证以 GitHub API 状态 + main 上的 CI 运行结果为权威，**不**以本地 pytest 为权威（main 上 CI 用 Python 3.11，本地是 3.9）。

## 0. 前置（环境校验）

```bash
gh auth status 2>&1 | grep -E "(zensgit|logged in)"
gh api repos/zensgit/cad-ml-platform/branches/main/protection \
  --jq '{required_approving_review_count: .required_pull_request_reviews.required_approving_review_count,
         enforce_admins: .enforce_admins.enabled}'
```

期望：
- `gh auth status` 显示 logged in
- `required_approving_review_count: 1`
- `enforce_admins: true`

若 `required_approving_review_count` 是 0 或 `enforce_admins=false`，**停下来**——意味着 protection 被人改过，先查 audit log。

## 1. Pre-merge 状态校验（每个 PR 都跑）

```bash
PR=468  # 重复对 471, 472
gh pr view "$PR" --json baseRefName,headRefName,mergeStateStatus,mergeable,reviewDecision \
  --jq '{base: .baseRefName, head: .headRefName, mergeStateStatus, mergeable, reviewDecision}'
```

期望（**所有 3 个 PR**）：

```json
{
  "mergeStateStatus": "CLEAN",
  "mergeable": "MERGEABLE",
  "reviewDecision": "APPROVED"
}
```

**任一字段不符 → 不要 merge**。失败矩阵：

| 字段 | 错误值 | 含义 | 处置 |
| --- | --- | --- | --- |
| `mergeStateStatus` | `BLOCKED` | 缺 review 或失败 check | 等 approve / 修 check |
| `mergeStateStatus` | `UNSTABLE` | 有 non-required 失败 check | 看 check 是否影响合并政策 |
| `mergeStateStatus` | `DIRTY` | 有合并冲突 | `gh pr edit --base main` + 解决冲突 |
| `mergeable` | `CONFLICTING` | 同上 | 同上 |
| `reviewDecision` | `REVIEW_REQUIRED` | 缺 approval | 等非 author approve |
| `reviewDecision` | `CHANGES_REQUESTED` | 有 changes requested | 处理 review comments，re-request |

## 2. Step A — 合 #468 验证

### 2.1 合入前

```bash
gh pr view 468 --json statusCheckRollup \
  --jq '[.statusCheckRollup[] | {name: (.name // .context), state: (.conclusion // .status // .state)}] | group_by(.state) | map({state: .[0].state, count: length})'
```

期望：所有 state 是 `SUCCESS` 或 `SKIPPED`，无 `FAILURE`。

### 2.2 合入

```bash
gh pr merge 468 --merge --delete-branch=false
echo "merge_exit=$?"
```

期望 `merge_exit=0`。

### 2.3 合入后立即校验

```bash
# main 上新 HEAD commit
git fetch origin main
git log origin/main -3 --oneline
# 期望第一行：Merge pull request #468 from ...

# PR 状态
gh pr view 468 --json state,mergedAt,mergeCommit --jq .
# 期望：state=MERGED, mergedAt=<时间戳>, mergeCommit.oid=<SHA>
```

### 2.4 触发的 main CI

```bash
# 等 60s 让 push event 触发所有 workflow
sleep 60
gh run list --branch main --limit 15 --created ">$(date -u -v-5M +%FT%TZ)" \
  --json workflowName,conclusion,status,databaseId \
  --jq '.[] | "\(.workflowName) \(.status) \(.conclusion // "-") \(.databaseId)"'
```

期望可见的 workflow 名（按之前观察的注册集合）：
- `CI`
- `CI Tiered Tests`
- `Governance Gates`
- `Code Quality` ← **首次跑**
- `Evaluation Report`
- `Self-Check`
- `Stress and Observability Checks`
- `Security Audit` / `SBOM` / `Action Pin Guard` 等多个

跟踪关键的 6 个：

```bash
for wf in "CI" "CI Tiered Tests" "Governance Gates" "Code Quality" "Evaluation Report" "Self-Check"; do
  run=$(gh run list --workflow "$wf" --branch main --limit 1 --json databaseId,conclusion,status \
        --jq '.[0] | "\(.databaseId) \(.status) \(.conclusion)"')
  echo "$wf -> $run"
done
```

最终每行期望 `<id> completed success`。

## 3. Step B — Retarget #471 + 合入验证

### 3.1 Retarget

```bash
gh pr edit 471 --base main
# 5–10 秒后再查
sleep 10
gh pr view 471 --json baseRefName,mergeStateStatus,mergeable --jq .
```

期望 `baseRefName: "main"`，`mergeStateStatus` 通常会变成 `BLOCKED`（要等新的 CI），mergeable 仍 `MERGEABLE`。

### 3.2 等 CI

```bash
gh pr checks 471 --watch
echo "checks_exit=$?"
```

期望 `checks_exit=0`。期间若有失败：

```bash
gh pr checks 471 --json name,conclusion,state \
  --jq '[.[] | select((.conclusion // .state) == "FAILURE") | .name]'
```

failed list 非空 → 看 log，多半是 retarget 后的 diff 与最新 main 冲突或 lint 漂移。

### 3.3 合入

```bash
gh pr merge 471 --merge --delete-branch=false
```

### 3.4 合入后

```bash
git fetch origin main
git log origin/main -3 --oneline
# 期望：Merge pull request #471 from ...
```

## 4. Step C — Retarget #472 + 合入验证

同 §3，但 PR=472，最后 `--delete-branch=true`：

```bash
gh pr edit 472 --base main
gh pr checks 472 --watch
gh pr merge 472 --merge --delete-branch=true
```

合入后：

```bash
git fetch origin main
git log origin/main -5 --oneline
# 期望前 3 行是 3 个 Merge pull request #468/471/472 (顺序)
```

## 5. brep-golden-eval.yml 注册验证

```bash
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)")'
```

期望：

```json
{"name": "B-Rep Golden Eval (OCC)", "state": "active"}
```

若 `state` 不在或 `disabled_manually`：

```bash
gh workflow enable brep-golden-eval.yml
# 再查一次
```

### 5.1 Smoke run

```bash
gh workflow run brep-golden-eval.yml --ref main
# 等几秒拿 run id
sleep 10
RUN_ID=$(gh run list --workflow brep-golden-eval.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
echo "smoke_run=$RUN_ID"
gh run watch "$RUN_ID" --exit-status
echo "smoke_exit=$?"
```

期望 `smoke_exit=0`。

### 5.2 Smoke artifact 校验

```bash
gh run download "$RUN_ID" --name brep-golden-eval
ls brep_golden_manifest/
cat brep_golden_manifest/summary.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['status'] == 'insufficient_release_samples', f\"unexpected status {d['status']}\"
assert d['release_eligible_count'] == 0, f\"example manifest should have 0 release-eligible, got {d['release_eligible_count']}\"
print('ok: example manifest smoke success')
"
```

期望 `ok: example manifest smoke success`。

## 6. Code Quality workflow 首跑结果分析

这是堆叠链合入解锁的**最高风险**项——之前从未在堆叠分支跑过。

```bash
CQ_RUN=$(gh run list --workflow code-quality.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view "$CQ_RUN" --json conclusion,jobs \
  --jq '{conclusion, failed_jobs: [.jobs[] | select(.conclusion == "failure") | .name]}'
```

### 6.1 若 success

记录此 run id 作为 baseline，回收 hand-off §2 中"accepted gap"标注。

### 6.2 若 failure

**预期最有可能的失败**：

| 失败 | 根因 | 修法 PR |
| --- | --- | --- |
| `Type Checking (mypy)` | Phase 5 新增 `decision_service.py` strict-mode 警告 | `chore(types): satisfy mypy strict on decision_service` |
| `Lint & Format` | 14 个新 `src/core/vector_*.py` 文件 isort 漂移 | `chore(lint): isort vector helper modules` |
| `Documentation Coverage` | 新增 18 个 module 没 module docstring | `chore(docs): module docstrings for vector helpers` |
| `Dead Code Detection` | facade 中 re-export 被认为 unused | 加 `__all__` 或 `# noqa: dead-code` 注释 |

每个失败开**独立 PR**（避免 batch revert）。

```bash
gh run view "$CQ_RUN" --log-failed | head -150
```

提取关键失败行。

## 7. Dependabot PRs 处置验证

逐个 close：

```bash
for pr in 389 391 392 393 394 469 470; do
  gh pr close $pr --comment "Blocked by Action Pin Guard policy (SHA pin required). Will re-roll via dependabot config follow-up: replace version tags with locked SHAs."
done

# 验证
gh pr list --state open --base main --author app/dependabot --json number --jq length
# 期望：0
```

## 8. 整体 Stage 1 出门验证（一次性总验）

```bash
# A) main 上 3 个 Merge commit 都在
git fetch origin main
git log origin/main --oneline | head -5 | grep -E "Merge pull request #(468|471|472)" | wc -l
# 期望：3

# B) 6 个真实门禁 workflow 全绿（取每个 workflow 最新 main 上的 run）
for wf in "CI" "CI Tiered Tests" "Governance Gates" "Code Quality" "Evaluation Report" "Self-Check"; do
  c=$(gh run list --workflow "$wf" --branch main --limit 1 --json conclusion --jq '.[0].conclusion')
  printf "%-25s %s\n" "$wf" "$c"
done | tee /tmp/stage1_gate_state.txt
grep -v "success" /tmp/stage1_gate_state.txt | wc -l
# 期望：0

# C) brep-golden-eval 注册 + smoke 过
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)") | .state'
# 期望：active

# D) 堆叠分支已删除
git branch -r | grep -E "phase3-vectors-(crud-router|list-router|batch-similarity-router)"
# 期望：仅 batch-similarity-router 可能存留若 --delete-branch=true 未在 Step C 用；其余应空

# E) dependabot 7 个 PR 处置完
gh pr list --state open --base main --author app/dependabot --json number --jq length
# 期望：0
```

**出门总条件**：A=3 ∧ B=0 ∧ C=active ∧ D 空 ∧ E=0。

## 9. 回滚预案

| 阶段 | 回滚命令 | 影响 |
| --- | --- | --- |
| Step A 后发现 #468 有问题 | `gh pr create --base main --head revert-468`（手动 revert merge commit） | main 上多两层 commit |
| Step B 后发现 #471 有问题 | 同上 | 同上 |
| Step C 后发现 #472 有问题 | 同上 | 整链需重做 |
| 全链都崩 | `pre-split-backup-20260515` tag 仍可重置堆叠分支（**不能** 重置 main） | 堆叠侧重做 |

**绝对禁止**：`git push --force origin main` — `allow_force_pushes: false` 也禁了这条路。

## 10. 失败模式快速参考

| 现象 | 诊断命令 | 处置 |
| --- | --- | --- |
| `gh pr merge` 返回 `merge state is not clean` | `gh pr view <N> --json mergeStateStatus` | 看 §1 表格 |
| Step B retarget 后 #471 出 `CONFLICTING` | `git fetch && git log origin/main..origin/<branch>` | 手动 rebase |
| Code Quality 红 | `gh run view --log-failed` | 看 §6.2 表格 |
| brep-golden-eval `state: disabled_manually` | `gh workflow enable brep-golden-eval.yml` | 重新 enable |
| Smoke artifact 含 `errors: [...]` 非空 | 看 errors 内容 | 多半是 example manifest 文件改过；恢复原版 |

## 11. 验证日志归档

```bash
mkdir -p reports/stage1
gh run list --branch main --created ">$(date -u -v-1H +%FT%TZ)" \
  --json workflowName,conclusion,databaseId,createdAt > reports/stage1/post_merge_runs.json
git log origin/main --oneline -10 > reports/stage1/main_tip.log
echo "stage1_completed_at=$(date -u +%FT%TZ)" > reports/stage1/done.txt
```

`reports/stage1/` 不在 `.gitignore` `reports/benchmark/` 范围内，可作为 artifact 留存。
