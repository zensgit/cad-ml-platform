# CAD ML Forward Roadmap — Stages 0 / 1 / 3 Final Verification

Date: 2026-05-18
Companion: `CAD_ML_FORWARD_ROADMAP_STAGE0_1_3_FINAL_DEVELOPMENT_20260518.md`

**Principle**: 验证以 GitHub API 状态 + `gh run view <id> --json conclusion` 为权威；本地 `py_compile`/`git log` 仅作 sanity。

## 0. 单点权威记录（认这一行）

| 维度 | 数据 |
| --- | --- |
| main HEAD | `b026fd8b` |
| `enforce_admins` 终态 | `true`（已 API 验证） |
| `required_approving_review_count` | `1`（未动） |
| `allow_force_pushes` | `false`（未动） |

## 1. main 上的 merge commit 链验证

```bash
git fetch origin main
git log origin/main --oneline -10
```

期望前 5 行包含（按时间倒序）：

```
b026fd8b  Merge pull request #474 ...
f2553f9d  Merge pull request #473 ...
01f12b87  Merge pull request #472 ...
3c5399bb  Merge pull request #471 ...
968e5fc8  Merge pull request #468 ...
```

5 个 Merge commit 都在 → Stage 1 + Stage 1 follow-up + Stage 3 全部合入。

## 2. branch protection 终态验证

```bash
gh api repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins \
  --jq '.enabled'
# 期望：true
```

```bash
gh api repos/zensgit/cad-ml-platform/branches/main/protection \
  --jq '{required_approving_review_count: .required_pull_request_reviews.required_approving_review_count,
         enforce_admins: .enforce_admins.enabled,
         allow_force_pushes: .allow_force_pushes.enabled}'
# 期望：{1, true, false}
```

## 3. 6 大门禁 + brep smoke first-run 验证

| Workflow | Run ID | 期望 conclusion |
| --- | --- | --- |
| CI | 26011196551 | success |
| CI Tiered Tests | 26011196580 | success |
| Governance Gates | 26008774788 | success |
| Code Quality | 26008774795 | success（hand-off accepted gap 解除） |
| Evaluation Report | 26008774768 | success |
| Self-Check | 26008774767 | success |
| B-Rep Golden Eval (OCC) smoke | 26008810573 | success |

```bash
for id in 26011196551 26011196580 26008774788 26008774795 26008774768 26008774767 26008810573; do
  c=$(gh run view "$id" --json conclusion --jq '.conclusion')
  printf "%-12s %s\n" "$id" "$c"
done
# 全部期望 "success"
```

## 4. Stage 0 — workflow-file-health fallback 行为

### 4.1 helper 单测（5 case parametric）

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.ci.check_workflow_file_issues import _is_missing_workflow_on_ref_error
cases = {
    'HTTP 404: workflow .github/workflows/brep-golden-eval.yml not found on the default branch (https://api.github.com/...)': True,
    'could not find workflow file, try specifying a different ref': True,
    'workflow was not found on the requested ref': True,
    'failed to log in: HTTP 401': False,
    'some unrelated error': False,
}
for msg, want in cases.items():
    got = _is_missing_workflow_on_ref_error(msg)
    assert got == want, f'{msg!r} -> got {got}, want {want}'
print('ok: all 5 cases match')
"
# 期望：ok: all 5 cases match
```

### 4.2 stress-tests workflow caller 验证

```bash
grep -n 'check_workflow_file_issues.py' .github/workflows/stress-tests.yml
# 期望：--mode auto（NOT --mode gh）

python3 -c "
import yaml
with open('.github/workflows/stress-tests.yml') as f:
    doc = yaml.safe_load(f)
for step in doc['jobs']['workflow-file-health']['steps']:
    if step.get('name') == 'Validate workflow file health via GitHub parser':
        assert '--mode auto' in step['run'], 'caller mode regressed'
        assert '--mode gh' not in step['run'], 'caller still has --mode gh'
print('ok')
"
```

### 4.3 stress-tests workflow 集成行为（PR #472 历史 run）

最初 run 25954579734 — failure（fallback 不触发）；后续 run 26008211481 — success（fallback 触发）；当前 main 上 stress-tests.yml 工作流 push 触发首跑应继续 success。

## 5. Stage 1 — 治理 cycle 审计

```bash
# 三次 cycle 都通过 GitHub API audit；本机查看（需 admin）
gh api 'repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins' --jq '.enabled'
# 期望：true（终态）

# main 上没有遗留的方案 C 临时提交
git log origin/main --oneline | grep -iE "(temp|wip|stash|todo-revert)" | head -5
# 期望：无输出
```

## 6. Stage 1 — `brep-golden-eval.yml` 已注册

```bash
gh workflow list --all --json name,state \
  --jq '.[] | select(.name == "B-Rep Golden Eval (OCC)")'
# 期望：{"name": "B-Rep Golden Eval (OCC)", "state": "active"}
```

## 7. Stage 3 — 5 个新单测落地验证

```bash
# 文件存在
for f in \
  tests/unit/test_decision_contract_schema.py \
  tests/unit/test_decision_service_evidence.py \
  tests/unit/test_model_readiness_registry_cache_invalidation.py \
  tests/unit/test_forward_scorecard_metric_status_truth_table.py \
  tests/unit/test_forward_scorecard_missing_component_blocked.py; do
  [ -f "$f" ] && echo "OK $f" || echo "MISSING $f"
done

# py_compile clean
for f in tests/unit/test_decision_contract_schema.py \
         tests/unit/test_decision_service_evidence.py \
         tests/unit/test_model_readiness_registry_cache_invalidation.py \
         tests/unit/test_forward_scorecard_metric_status_truth_table.py \
         tests/unit/test_forward_scorecard_missing_component_blocked.py; do
  python3 -m py_compile "$f" && echo "PY-OK $f" || echo "PY-FAIL $f"
done
```

期望 5/5 OK + 5/5 PY-OK。

## 8. Stage 3 — 关键 invariant 在 CI 上验证

PR #474 PR-level CI（合入前）：failing=0

合入后 main 上：
- CI run 26018132929 — 验证 `tests (3.10)` / `tests (3.11)` 含新测试
- CI Tiered Tests run 26018132993 — 验证 `unit-tier` 含新测试
- Code Quality run 26018132996 — 验证 type/lint 仍 OK

```bash
for id in 26018132929 26018132993 26018132996; do
  c=$(gh run view "$id" --json conclusion --jq '.conclusion')
  printf "%-12s %s\n" "$id" "$c"
done
# 全部期望 "success"
```

## 9. Stage 3 — 不破坏现有测试矩阵

```bash
# 没动生产代码
git show --stat 41971faf -- 'src/**' | wc -l
# 期望：0（commit 41971faf 不包含 src/ 文件）

git show --stat 41971faf -- 'tests/**' | grep -E '^\s+tests/'
# 期望：仅 5 个新 test 文件，无修改现有测试
```

## 10. 一次性总验脚本

```bash
#!/bin/bash
set -e
fails=0

# 10.1 main HEAD 含 5 个目标 merge commit
git fetch origin main >/dev/null 2>&1
expected=("968e5fc8" "3c5399bb" "01f12b87" "f2553f9d" "b026fd8b")
for sha in "${expected[@]}"; do
  if ! git log origin/main --oneline | head -10 | grep -q "$sha"; then
    echo "MISS: $sha not in main tip"
    fails=$((fails+1))
  fi
done

# 10.2 enforce_admins 已恢复
enforced=$(gh api repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled')
[ "$enforced" = "true" ] || { echo "ENF: enforce_admins=$enforced"; fails=$((fails+1)); }

# 10.3 brep workflow 注册
brep_state=$(gh workflow list --all --json name,state --jq '.[] | select(.name=="B-Rep Golden Eval (OCC)") | .state')
[ "$brep_state" = "active" ] || { echo "BREP: state=$brep_state"; fails=$((fails+1)); }

# 10.4 5 个 Stage 3 test 文件存在
for f in tests/unit/test_decision_contract_schema.py \
         tests/unit/test_decision_service_evidence.py \
         tests/unit/test_model_readiness_registry_cache_invalidation.py \
         tests/unit/test_forward_scorecard_metric_status_truth_table.py \
         tests/unit/test_forward_scorecard_missing_component_blocked.py; do
  [ -f "$f" ] || { echo "MISSING: $f"; fails=$((fails+1)); }
done

# 10.5 helper regex 5 case
python3 -c "
from scripts.ci.check_workflow_file_issues import _is_missing_workflow_on_ref_error
assert _is_missing_workflow_on_ref_error('HTTP 404: ... not found on the default branch')
assert _is_missing_workflow_on_ref_error('could not find workflow file, try specifying a different ref')
assert _is_missing_workflow_on_ref_error('workflow was not found on the requested ref')
assert not _is_missing_workflow_on_ref_error('failed to log in: HTTP 401')
assert not _is_missing_workflow_on_ref_error('some unrelated error')
" || { echo "HELPER: regex regression"; fails=$((fails+1)); }

echo "---"
[ "$fails" -eq 0 ] && echo "ALL GREEN" || echo "FAILED $fails item(s)"
```

期望：`ALL GREEN`。

## 11. main 上 Stage 3 新测试首跑结果（已验证）

PR #474 合入后 main 上的新 CI run：

| Run ID | Workflow | 结果 |
| --- | --- | --- |
| 26018132929 | CI | ✅ success |
| 26018132993 | CI Tiered Tests | ✅ success |
| 26018132996 | Code Quality | ✅ success |

5 个新单测在 main 上首跑全过：
- `test_decision_contract_schema.py`
- `test_decision_service_evidence.py`
- `test_model_readiness_registry_cache_invalidation.py`
- `test_forward_scorecard_metric_status_truth_table.py`
- `test_forward_scorecard_missing_component_blocked.py`

Stage 3 闭环。release infrastructure 加固层在主干生效。

## 12. 异常 / 反例验证

### 12.1 self-approve 被拒（应该）

```bash
gh pr review <new-pr> --approve --body "self"
# 期望：API error 422 / forbidden（author 不能 approve 自己）
```

### 12.2 admin merge 不带 enforce_admins toggle（应该失败）

```bash
gh pr merge <pr> --merge --admin
# 期望：merge state is not clean / blocked by branch protection
```

### 12.3 force push（应该失败）

```bash
git push --force origin main
# 期望：remote rejected — non-fast-forward / protected branch
```

如以上任一未按预期失败，说明 protection 配置有漏洞，停下查 audit log。

## 13. 验证日志归档

```bash
mkdir -p reports/stage0_1_3
git log origin/main --oneline -15 > reports/stage0_1_3/main_tip.log
gh api repos/zensgit/cad-ml-platform/branches/main/protection \
  > reports/stage0_1_3/main_protection.json
echo "verified_at=$(date -u +%FT%TZ)" > reports/stage0_1_3/done.txt
```

`reports/stage0_1_3/` 不在 gitignored 路径，可作为长期 artifact 留存。
