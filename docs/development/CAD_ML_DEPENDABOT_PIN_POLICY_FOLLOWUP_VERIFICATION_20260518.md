# Dependabot Pin Policy — Follow-up Verification

Date: 2026-05-18
Companion: `CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_DEVELOPMENT_20260518.md`
Tracking issue: #476

## 0. 验证范围

本 follow-up 只验证三件事：

1. Issue #476 已起且内容完整
2. 7 个旧 dependabot PR 已 close（人工执行后）
3. Development + Verification MD 已合入 main

**不**验证：永久修法（dependabot.yml 改造）— 那是 issue #476 自己的范畴。

## 1. Issue #476 完整性

```bash
gh issue view 476 --json title,state,body --jq '{title, state, body_len: (.body | length)}'
```

期望：
- `title`: "dependabot config: emit SHA-pinned PRs to satisfy Action Pin Guard"
- `state`: "OPEN"
- `body_len`: ≥ 1500（确保不是空 issue）

```bash
gh issue view 476 --json body --jq '.body' | grep -cE "^(## Why|## What|## Scope|## Acceptance|## References)"
```

期望：5（5 个章节齐全）

## 2. 7 个旧 dependabot PR 状态

```bash
gh pr list --state open --base main --author app/dependabot --json number --jq length
```

期望（在人工执行 §3.1 命令后）：`0`

或者：仍是 7，说明 close 还未执行——查 §3 修法。

逐个 verify state：

```bash
for pr in 389 391 392 393 394 469 470; do
  s=$(gh pr view $pr --json state --jq '.state')
  printf "#%d -> %s\n" "$pr" "$s"
done
```

期望全部 `CLOSED`。

## 3. 人工 close 命令

如 §2 显示仍有 open dependabot PR：

```bash
for pr in 389 391 392 393 394 469 470; do
  gh pr close $pr --comment "Closing per Stage 1 disposition (docs/development/CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md §3.3): blocked by repo's Action Pin Guard policy. Tracking re-roll in #476."
done
```

执行后回到 §2 重新验证。

## 4. main 上 MD 落地验证

合入 main 后：

```bash
git fetch origin main
git ls-files docs/development/ | grep -E "DEPENDABOT_PIN_POLICY_FOLLOWUP_(DEVELOPMENT|VERIFICATION)_20260518.md"
```

期望：

```
docs/development/CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_DEVELOPMENT_20260518.md
docs/development/CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_VERIFICATION_20260518.md
```

## 5. main CI 状态不退化

合入 follow-up MD 后（cycle 5），等下列 workflow first-run 全绿：

```bash
gh run list --branch main --limit 6 --created ">$(date -u -v-3M +%FT%TZ)" \
  --json workflowName,conclusion,status,databaseId \
  --jq '.[] | select(.workflowName | test("^(CI|CI Tiered Tests|Code Quality)$")) | "\(.status) \(.conclusion // "-") \(.workflowName) (\(.databaseId))"'
```

期望：3 个 workflow 全部 `completed success`（docs-only change 不应破任何 CI）。

## 6. 治理边界未被弱化

```bash
gh api repos/zensgit/cad-ml-platform/branches/main/protection \
  --jq '{required_approving_review_count: .required_pull_request_reviews.required_approving_review_count,
         enforce_admins: .enforce_admins.enabled,
         allow_force_pushes: .allow_force_pushes.enabled}'
```

期望（与 Stage 1 完成时一致）：

```json
{
  "required_approving_review_count": 1,
  "enforce_admins": true,
  "allow_force_pushes": false
}
```

## 7. 与 Stage 2a 的兼容性

**Issue #476 实施前**：不应试图升级 `mamba-org/setup-micromamba`。验证：

```bash
grep -nE "setup-micromamba@" .github/workflows/brep-golden-eval.yml
```

期望命中：`uses: mamba-org/setup-micromamba@4b9113af4fba0e9e1124b252dd6497a419e7396d`（即 1.5.8-0 的 SHA pin），**未升 3.0.0**。

## 8. Issue #476 实施时的额外验证（仅在真实施时）

如果之后真有人去实现 issue #476，**必须**在那次 PR 中加入：

```bash
# 把 .github/workflows/brep-golden-eval.yml 的 setup-micromamba SHA 改完后：
gh workflow run brep-golden-eval.yml --ref <fix-branch>
gh run watch  # 必须 success
```

否则 Stage 2a runtime workflow 会被打挂。

## 9. 一次性总验脚本

```bash
#!/bin/bash
set -e
fails=0

# 9.1 Issue #476 存在且 OPEN
state=$(gh issue view 476 --json state --jq '.state' 2>/dev/null || echo "MISS")
[ "$state" = "OPEN" ] || { echo "ISSUE: state=$state"; fails=$((fails+1)); }

# 9.2 0 个 dependabot PR open
open_count=$(gh pr list --state open --base main --author app/dependabot --json number --jq length)
[ "$open_count" -eq 0 ] || { echo "DEPENDABOT: $open_count still open"; fails=$((fails+1)); }

# 9.3 两份 MD 存在
for f in \
  docs/development/CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_DEVELOPMENT_20260518.md \
  docs/development/CAD_ML_DEPENDABOT_PIN_POLICY_FOLLOWUP_VERIFICATION_20260518.md; do
  [ -f "$f" ] || { echo "MISSING: $f"; fails=$((fails+1)); }
done

# 9.4 治理保留
enforce=$(gh api repos/zensgit/cad-ml-platform/branches/main/protection/enforce_admins --jq '.enabled')
[ "$enforce" = "true" ] || { echo "ENF: enforce_admins=$enforce"; fails=$((fails+1)); }

# 9.5 micromamba 仍 1.5.8-0 SHA
grep -q "setup-micromamba@4b9113af4fba0e9e1124b252dd6497a419e7396d" .github/workflows/brep-golden-eval.yml \
  || { echo "MMAMBA: pin changed"; fails=$((fails+1)); }

echo "---"
[ "$fails" -eq 0 ] && echo "ALL GREEN" || echo "FAILED $fails item(s)"
```

期望：`ALL GREEN`。

## 10. 异常处理

| 现象 | 诊断 | 处置 |
| --- | --- | --- |
| `gh pr close` 拒绝（permissions） | author/maintainer 权限不足 | 用 admin token 跑；或在 GitHub UI 手动 close |
| Issue #476 已被自动关闭 | bot 误判 / 重复 issue | 重新 open 并 link 本文档 |
| Dependabot 重发新 PR 立刻又触发 Action Pin Guard | 永久修法未落地 | 关闭新 PR；issue #476 优先级提升 |
| main CI 在 docs-only PR 上失败 | check_workflow_action_pins 误以为 doc 中提到 `@v5` 是真的 use 语句 | 看 fail 详情；若 false positive 在 pin guard 中加 `.md` exclusion |
