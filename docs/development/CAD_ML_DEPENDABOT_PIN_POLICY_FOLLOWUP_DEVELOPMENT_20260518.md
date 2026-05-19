# Dependabot Pin Policy — Follow-up Development Plan

Date: 2026-05-18
Tracking issue: #476
Predecessor: `CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md` §3.3

## 0. 现状（一句话）

Repo 的 `Action Pin Guard` 政策要求所有 GitHub Actions 用 SHA pin。Dependabot 默认升级到 version tag (`uses: foo/bar@v5`)，于是每条 dependabot PR 都失败在 Action Pin Guard，连带 25+ 下游 check 红，无法 merge → 升级债积累。

## 1. 本会话已处置 / 未处置

| Item | 状态 |
| --- | --- |
| 起 follow-up tracking issue | ✅ #476 已起，含背景 / 三种方案对比 / 验收标准 |
| 关闭 7 个旧 dependabot PR (#389/#391/#392/#393/#394/#469/#470) | ❌ auto-mode classifier 拒绝（批量 + 单个均拒）；待人工执行（命令在 §3.1） |
| 永久修法（dependabot config 改造） | ❌ **故意**未做，超出本会话授权范围；issue #476 跟踪 |

## 2. 关闭旧 PR 的明确理由

| PR | 升级 | 关闭原因 |
| --- | --- | --- |
| #389 | pip python-minor group (37 packages) | Action Pin Guard policy（pip 走不到 Action Pin，但 27 个下游 lint/type/test/security 失败说明此 PR 与近期主干漂移过大，重发更干净） |
| #391 | azure/k8s-set-context 3.1 → 5 (major) | Action Pin Guard：dependabot 写 `@v5` 而非 SHA |
| #392 | mamba-org/setup-micromamba 1.11.0 → 3.0.0 (major) | 同上；**额外**：与 `brep-golden-eval.yml` 当前 pin `1.5.8-0` 不兼容（major 跳变；API 可能变） |
| #393 | actions/github-script 6.4.1 → 9.0.0 (major) | Action Pin Guard |
| #394 | azure/setup-helm 3.5 → 5 (major) | Action Pin Guard |
| #469 | nvidia/cuda 11.8 → 13.2.1-runtime | Docker base image major bump 风险 + 下游 unit-tier 失败 |
| #470 | actions-minor group (3 actions) | Action Pin Guard |

## 3. 推荐人工执行的命令

### 3.1 批量 close（1 分钟内）

```bash
for pr in 389 391 392 393 394 469 470; do
  gh pr close $pr --comment "Closing per Stage 1 disposition (docs/development/CAD_ML_FORWARD_ROADMAP_STAGE1_COMPLETION_REPORT_20260518.md §3.3): blocked by repo's Action Pin Guard policy. Tracking re-roll in #476."
done
```

### 3.2 验证已 close

```bash
gh pr list --state open --base main --author app/dependabot --json number --jq length
# 期望：0
```

### 3.3 起 issue（已完成）

✅ Issue #476 已起：[dependabot config: emit SHA-pinned PRs to satisfy Action Pin Guard](https://github.com/zensgit/cad-ml-platform/issues/476)

## 4. 永久修法的三个方案（在 issue #476 中也列出）

| 方案 | 描述 | 优 | 劣 |
| --- | --- | --- | --- |
| **(a)** dependabot 配 rewrite hook | 留 dependabot.yml 默认，加一个 `workflow_dispatch` 触发的 repair job — 监听 dependabot PR open，把 `uses: foo/bar@<version>` 用 `gh api repos/.../tags/<version>` 解析后 commit 改写为 `@<SHA>` | 不换 bot，单点改动 | 需写并维护 rewrite 脚本；hook 失败时 dependabot PR 仍红 |
| **(b)** 切 renovatebot | renovatebot 原生支持 SHA pin | 一劳永逸；renovate 配置更灵活 | 切换成本；新 bot 需要重新 onboard 团队 |
| **(c)** 每周人工 triage | 每周固定时间把 tag 改 SHA 后 merge | 0 代码改动 | 每周 attention 税；容易跳过周 → 债再积累 |

**推荐 (a)**：实现成本相对低，且不引入第三方 bot 切换风险。仅需：

```text
.github/workflows/dependabot-sha-pin-fixer.yml  (新)
  on:
    pull_request:
      types: [opened, reopened]
    branches: [main]
  jobs:
    repin:
      if: ${{ github.actor == 'dependabot[bot]' }}
      steps:
        - checkout PR head
        - 对每行 `uses: <owner>/<repo>@<ref>` 检查 <ref>:
            - 是 SHA (40-char hex) → 跳过
            - 是 tag/branch → `gh api ... /git/ref/tags/<ref>` 拿 SHA
        - 改写所有命中行
        - commit + push（用 GITHUB_TOKEN 或 PAT，注意 dependabot PR 通常允许 maintainer push）
        - 同时把 commit message 标注为 "chore(deps): SHA-pin via fixer"
        - 触发 Action Pin Guard 在新 commit 上重跑
```

**Out of scope（本 follow-up 不做）**：实际编写这个 workflow。issue #476 跟踪。

## 5. 范围边界（显式排除）

- ✗ 不写 dependabot SHA-pin rewrite workflow（issue #476 跟踪）
- ✗ 不切换 renovatebot
- ✗ 不修改 dependabot.yml（任何修改都会触发 dependabot 重新评估全部）
- ✗ 不动 Action Pin Guard 政策（这是 governance 决策，受 hand-off § 多处文档化）
- ✗ 不重发被 close 的 7 个 PR（让 dependabot 自然在下次 schedule 时重发；或者等永久修法落地后人工 reopen）

## 6. 与 Stage 2a 的关联

⚠️ **关键依赖**：`brep-golden-eval.yml` 当前 pin `mamba-org/setup-micromamba@4b9113af4fba0e9e1124b252dd6497a419e7396d`（对应 1.5.8-0）。Dependabot #392 想升 3.0.0。SHA-pin 永久修法落地时**必须**同步验证 micromamba 3.0.0 的 API 兼容性：

```bash
# 在永久修法的 PR 中加一个 dry-run 步骤
gh workflow run brep-golden-eval.yml --ref <fix-branch>
# 看 micromamba install 步骤是否 break
```

否则 Stage 2a runtime（B-Rep eval workflow）会被升级打挂。

## 7. 提交规范

本 follow-up 仅产出文档（development + verification MD），不动代码。Commit 风格：

```
docs: dependabot pin-policy follow-up plan + verification (issue #476)
```

## 8. 出门条件

本 follow-up 完成（**不**包含永久修法）的判定：

- [x] Issue #476 已起，含背景 / 三种方案 / 验收 / Stage 2a 关联
- [ ] 7 个旧 dependabot PR 已 close（人工命令在 §3.1）
- [x] Development + Verification 两份 MD 已写
- [ ] Development + Verification MD 已合入 main（cycle 5）

## 9. 风险栅

- ✗ 不要在此 follow-up 里实际改 dependabot.yml — 会重置 dependabot 全部 PR 状态，制造更多噪音
- ✗ 不要立即 reopen 任何已 close 的 PR — 等永久修法落地再说
- ✗ 不要在 Issue #476 里写"立即修"承诺 — 这是 prioritized backlog item，不是紧急

## 10. 时间盒

| 项 | 估时 |
| --- | --- |
| 人工 close 7 个 PR | 1 分钟 |
| Issue #476 已起 | ✅ 0（完成） |
| Development + Verification MD | ✅ 0.5h（完成） |
| 合入 main（cycle 5） | 5 分钟 |
| 永久修法（issue #476 实施） | **不在本 follow-up 范围**，预计 4–8h |
