# CAD ML — Session Final Verification

Date: 2026-05-18
Companion: `CAD_ML_SESSION_FINAL_DEVELOPMENT_20260518.md`

**Principle**: 验证以 GitHub API 状态 + `gh run view <id> --json conclusion` 为权威。任何"我看过了感觉对"不算验证。

## 0. 一行权威状态（验证完唯一保留的东西）

| 维度 | 值 |
|---|---|
| main HEAD | `66e50ec8` |
| 本会话合入数 | 5 个 PR + 1 个 issue |
| 5 次方案 C cycle 全部成功 | yes |
| `enforce_admins` 终态 | `true` |
| 7 个 dependabot PR 状态 | 仍 open（待人工 close）|

## 1. main 上 5 个新 merge commit 都在

```bash
git fetch origin main
git log origin/main --oneline | head -15 | grep -cE "Merge pull request #(468|471|472|473|474|475|477)"
```

期望：**7** （5 个 cycle = 7 个 PR merge commit；#468/#471/#472 同 cycle 但各自有自己的 merge commit）

具体 SHA 对照：

```
968e5fc8  #468  vectors crud router
3c5399bb  #471  vectors list router
01f12b87  #472  Phase 1-6 + Stage 0 hotfix
f2553f9d  #473  Stage 0 follow-up
b026fd8b  #474  Stage 3 hardening
1edcd307  #475  Stage 0+1+3 final docs
66e50ec8  #477  dependabot follow-up docs
```

## 2. 治理边界终态

```bash
gh api repos/zensgit/cad-ml-platform/branches/main/protection --jq '{
  required_approving_review_count: .required_pull_request_reviews.required_approving_review_count,
  enforce_admins: .enforce_admins.enabled,
  allow_force_pushes: .allow_force_pushes.enabled
}'
```

期望：

```json
{
  "required_approving_review_count": 1,
  "enforce_admins": true,
  "allow_force_pushes": false
}
```

## 3. main 上 7 个 CI workflow 全绿

### 3.1 第一次堆叠链合入后首跑

```bash
for id in 26011196551 26011196580 26008774788 26008774795 26008774768 26008774767 26008810573; do
  c=$(gh run view "$id" --json conclusion --jq '.conclusion')
  printf "%-12s %s\n" "$id" "$c"
done
```

期望 7 个全部 `success`。

### 3.2 Stage 3 加固后 3 个 workflow

```bash
for id in 26018132929 26018132993 26018132996; do
  c=$(gh run view "$id" --json conclusion --jq '.conclusion')
  printf "%-12s %s\n" "$id" "$c"
done
```

期望 3 个全部 `success`。

## 4. 5 个新 Stage 3 单测在 main 上

```bash
for f in \
  tests/unit/test_decision_contract_schema.py \
  tests/unit/test_decision_service_evidence.py \
  tests/unit/test_model_readiness_registry_cache_invalidation.py \
  tests/unit/test_forward_scorecard_metric_status_truth_table.py \
  tests/unit/test_forward_scorecard_missing_component_blocked.py; do
  [ -f "$f" ] && echo "OK $f" || echo "MISSING $f"
done
```

期望 5 个 OK。

## 5. 3 处 CI 修复在 main 上

```bash
# 5.1 helper regex 包含 3 个匹配分支
python3 -c "
from scripts.ci.check_workflow_file_issues import _is_missing_workflow_on_ref_error
assert _is_missing_workflow_on_ref_error('HTTP 404: ... not found on the default branch')
assert _is_missing_workflow_on_ref_error('could not find workflow file, try specifying a different ref')
assert _is_missing_workflow_on_ref_error('workflow was not found on the requested ref')
assert not _is_missing_workflow_on_ref_error('failed to log in: HTTP 401')
print('OK helper regex 3+2')
"

# 5.2 stress-tests caller mode
grep -q '\--mode auto' .github/workflows/stress-tests.yml && echo "OK caller mode" || echo "FAIL caller mode"

# 5.3 stress-tests assertion aligned
grep -q '"--mode auto" in run_script' tests/unit/test_stress_workflow_workflow_file_health.py \
  && echo "OK test assertion" || echo "FAIL test assertion"
```

期望 3 个全 OK。

## 6. Issue #476 完整性

```bash
gh issue view 476 --json title,state,body --jq '{title, state, body_len: (.body | length)}'
```

期望：
- `state`: `OPEN`
- `body_len` ≥ 1500
- 含 5 个章节: Why / What / Scope / Acceptance / References

```bash
gh issue view 476 --json body --jq '.body' | grep -cE "^(## Why|## What|## Scope|## Acceptance|## References)"
# 期望：5
```

## 7. 7 个 dependabot PR 状态

```bash
gh pr list --state open --base main --author app/dependabot --json number --jq length
```

期望：
- **若已人工 close**: 0
- **若未 close**: 7（待人工）

逐个详查：

```bash
for pr in 389 391 392 393 394 469 470; do
  s=$(gh pr view $pr --json state --jq '.state')
  printf "#%d -> %s\n" "$pr" "$s"
done
```

## 8. brep-golden-eval.yml 注册仍 active

```bash
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)") | .state'
```

期望：`active`

## 9. 反例验证（应该全部失败）

### 9.1 self-approve 必须被拒

```bash
gh pr review <any-open-pr> --approve
# 期望：422 / forbidden（author 不能 approve 自己）
```

### 9.2 非 cycle admin-merge 必须被拒

```bash
gh pr merge <any-open-pr> --merge --admin
# 期望：merge state is not clean / blocked by required reviews
```

### 9.3 force push main 必须被拒

```bash
git push --force origin main
# 期望：remote rejected — protected branch
```

任一未按预期失败 → 治理有漏洞，停下查 audit log。

## 10. 一次性总验脚本

```bash
#!/bin/bash
set -e
fails=0

# 10.1 main 头 7 个 merge commit 都在
git fetch origin main >/dev/null 2>&1
merge_count=$(git log origin/main --oneline | head -15 | grep -cE "Merge pull request #(468|471|472|473|474|475|477)")
[ "$merge_count" -eq 7 ] || { echo "MERGE: $merge_count/7 found"; fails=$((fails+1)); }

# 10.2 governance
enforce=$(gh api repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled')
[ "$enforce" = "true" ] || { echo "ENF: enforce_admins=$enforce"; fails=$((fails+1)); }

review_count=$(gh api repos/zensgit/cad-ml-platform/branches/main/protection \
  --jq '.required_pull_request_reviews.required_approving_review_count')
[ "$review_count" = "1" ] || { echo "REV: review_count=$review_count"; fails=$((fails+1)); }

# 10.3 7 个目标 workflow run 全绿
for id in 26011196551 26011196580 26008774788 26008774795 26008774768 26008774767 26008810573 26018132929 26018132993 26018132996; do
  c=$(gh run view "$id" --json conclusion --jq '.conclusion' 2>/dev/null)
  [ "$c" = "success" ] || { echo "RUN: $id=$c"; fails=$((fails+1)); }
done

# 10.4 5 个新单测文件
for f in \
  tests/unit/test_decision_contract_schema.py \
  tests/unit/test_decision_service_evidence.py \
  tests/unit/test_model_readiness_registry_cache_invalidation.py \
  tests/unit/test_forward_scorecard_metric_status_truth_table.py \
  tests/unit/test_forward_scorecard_missing_component_blocked.py; do
  [ -f "$f" ] || { echo "TEST: $f missing"; fails=$((fails+1)); }
done

# 10.5 helper regex 行为
python3 -c "
from scripts.ci.check_workflow_file_issues import _is_missing_workflow_on_ref_error
assert _is_missing_workflow_on_ref_error('HTTP 404: ... not found on the default branch')
assert _is_missing_workflow_on_ref_error('could not find workflow file, try specifying a different ref')
assert _is_missing_workflow_on_ref_error('workflow was not found on the requested ref')
assert not _is_missing_workflow_on_ref_error('failed to log in: HTTP 401')
" || { echo "REGEX: helper regression"; fails=$((fails+1)); }

# 10.6 stress-tests caller mode
grep -q '\--mode auto' .github/workflows/stress-tests.yml \
  || { echo "MODE: caller regressed"; fails=$((fails+1)); }

# 10.7 issue #476 open
state=$(gh issue view 476 --json state --jq '.state' 2>/dev/null)
[ "$state" = "OPEN" ] || { echo "ISSUE: 476 state=$state"; fails=$((fails+1)); }

# 10.8 brep workflow active
brep_state=$(gh workflow list --all --json name,state --jq '.[] | select(.name=="B-Rep Golden Eval (OCC)") | .state')
[ "$brep_state" = "active" ] || { echo "BREP: state=$brep_state"; fails=$((fails+1)); }

# 10.9 11 份新文档（除本文外）都在 main 上
docs=(
  CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_DEVELOPMENT_20260517.md
  CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_VERIFICATION_20260517.md
  CAD_ML_FORWARD_ROADMAP_NEXT_STAGES_TODO_20260517.md
  CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_DEVELOPMENT_20260517.md
  CAD_ML_FORWARD_ROADMAP_STAGE1_LAND_STACK_VERIFICATION_20260517.md
  CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md
  CAD_ML_FORWARD_ROADMAP_STAGE0_1_3_FINAL_DEVELOPMENT_20260518.md
  CAD_ML_FORWARD_ROADMAP_STAGE0_1_3_FINAL_VERIFICATION_20260518.md
  CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_DEVELOPMENT_20260518.md
  CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_VERIFICATION_20260518.md
)
for d in "${docs[@]}"; do
  [ -f "docs/development/$d" ] || { echo "DOC: $d missing"; fails=$((fails+1)); }
done

echo "---"
[ "$fails" -eq 0 ] && echo "ALL GREEN ($((${#docs[@]} + 5 + 10 + 1)) checks)" || echo "FAILED $fails item(s)"
```

期望：`ALL GREEN`（除非 7 个 dependabot PR 还 open，那是 §7 的人工 follow-up，不进总验）。

## 11. 验证日志归档

```bash
mkdir -p reports/session-final
git log origin/main --oneline -15 > reports/session-final/main_tip.log
gh api repos/zensgit/cad-ml-platform/branches/main/protection > reports/session-final/protection.json
gh issue view 476 --json title,state,body > reports/session-final/issue_476.json
gh pr list --state open --base main --author app/dependabot \
  --json number,title,state > reports/session-final/dependabot_open.json
date -u +%FT%TZ > reports/session-final/verified_at.txt
```

`reports/session-final/` 不在 gitignored 范围，可作长期 artifact。

## 12. 若本文档要合入 main（cycle 6）

需要再走一次方案 C cycle，且需要你明确授权（classifier 默认每个 cycle 都需 user 显式提及当前 PR 号）。命令模板：

```bash
gh api -X DELETE repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins
gh pr merge <new-pr> --merge --admin --delete-branch=true
gh api -X POST   repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins
gh api          repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled'
# 必须 = true
```

或者本文档 + companion 保留在本地，作为 session artifact（不进 main）。两种选择都可接受——本文档是 session 总结，不是 release-critical。
