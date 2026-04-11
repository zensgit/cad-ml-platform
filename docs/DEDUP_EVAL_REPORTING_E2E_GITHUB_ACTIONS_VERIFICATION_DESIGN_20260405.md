# Eval Reporting E2E GitHub Actions Verification — Design

日期：2026-04-05

## Scope

Batch 23A 只做真实 GitHub-hosted `Evaluation Report` 运行态验收，验证经过 Batch 22 consolidate 与后续最小 hotfix 收口后的 `eval reporting` 栈，是否已经在 `push` 到 `main` 的真实 run 中完成 end-to-end materialization。

## Qualifying Run Criteria

一个 qualifying run 必须同时满足：

1. Workflow = `Evaluation Report` (`evaluation-report.yml`)
2. Branch = `main`
3. Event = `push`
4. `deploy-pages` job 实际执行，不是 skipped
5. run 对应的 workflow 内容已包含 Batch 22 consolidate 以及后续 startup / history-validation hotfix

## Verified Run

最终 qualifying run 为：

| Field | Value |
|---|---|
| Run ID | `24066289833` |
| URL | `https://github.com/zensgit/cad-ml-platform/actions/runs/24066289833` |
| Event | `push` |
| Branch | `main` |
| Head SHA | `8d2dbb644f7c0a5e724217e3f41a8fff11594c90` |
| Head commit | `fix: skip generated eval history surfaces (#381)` |
| Overall conclusion | `success` |

## Run-Level Findings

| Job | Database ID | Conclusion | Evidence |
|---|---:|---|---|
| `Run Evaluation and Generate Report` | `70192979470` | `success` | evaluate-side artifact generation, history validation, status-check posting step all completed |
| `Deploy Report to GitHub Pages` | `70193123603` | `success` | Pages deploy + post-deploy eval reporting surfaces all materialized |

关键步骤结论：

- `Validate history with JSON Schema`: `success`
- `Post Eval Reporting status check`: step 本身 `success`，但日志记录 `Status check skipped (fail-soft): Resource not accessible by integration`
- `Deploy to GitHub Pages`: `success`
- `Consolidated eval reporting deploy-pages summary`: `success`

## Artifact-Level Findings

该 run 共生成 `15` 个 artifacts，其中与收口后的 eval reporting 直接相关的保留面全部存在：

1. `eval-reporting-pages-1490`
2. `eval-reporting-public-index-1490`
3. `eval-reporting-dashboard-payload-1490`
4. `eval-reporting-webhook-delivery-request-1490`
5. `eval-reporting-webhook-delivery-result-1490`
6. `eval-reporting-release-draft-publish-result-1490`

同时仍保留 evaluate-side owner / stack artifacts：

1. `evaluation-report-1490`
2. `evaluation-interactive-report-1490`
3. `eval-reporting-stack-1490`
4. `eval-reporting-landing-1490`
5. `evaluation-history-1490`
6. `eval-reporting-stack-summary-1490`
7. `eval-reporting-release-summary-1490`
8. `hybrid-superpass-gate-1490`
9. `github-pages`

这与 Batch 16/17/18/19/20/21/22 收口后的 target architecture 一致，没有回流已删除 surface。

## Pages-Level Findings

`gh api repos/zensgit/cad-ml-platform/pages` 返回：

- `html_url = https://zensgit.github.io/cad-ml-platform/`
- `build_type = workflow`
- `source.branch = main`
- `source.path = /`
- `public = true`
- `https_enforced = true`

这说明：

1. Pages 已启用
2. `deploy-pages` job 已切到 GitHub Pages workflow 模式
3. 真实 `push/main` run 已成功把 Pages-ready root 发布出去

## External Consumer Findings

### Success

- `Send notifications`: `success`
- `Comment PR with results`: `skipped`
  - 原因不是失败，而是本次是 `push/main` run，没有 PR comment 目标

### Recorded Fail-Soft Behavior

- `Post Eval Reporting status check` step 运行成功，但日志明确记录：
  - `Status check skipped (fail-soft): Resource not accessible by integration`
- 对同一 `head sha` 查询 commit status API 时：
  - `gh api repos/zensgit/cad-ml-platform/commits/8d2dbb644f7c0a5e724217e3f41a8fff11594c90/status`
  - 返回 `total_count = 0`

这说明独立 `Eval Reporting` commit status surface 没有真正 materialize 到 merge commit 上，但 workflow 设计本来就是 fail-soft，不会阻断主链 E2E。

## Interpretation

Batch 23A 的核心目标已经满足：

1. 找到了真实 qualifying `push/main` run
2. evaluate job 与 deploy-pages job 均成功
3. Pages 成功发布
4. 收口后的 5 个 post-deploy eval reporting surfaces 全部 materialize
5. consolidated deploy-pages summary 实际执行成功

唯一未完全 materialize 的外部面是独立 commit status surface；该问题已被真实记录，但不影响本轮对 eval reporting workflow 主线的 end-to-end 验收。

## What Was Not Done In This Batch

- 没有修改 `.github/workflows/evaluation-report.yml`
- 没有修改 `scripts/ci/*`
- 没有修改 `tests/unit/*`
- 没有把 `workflow_dispatch` 冒充为 full deploy-pages 验收

