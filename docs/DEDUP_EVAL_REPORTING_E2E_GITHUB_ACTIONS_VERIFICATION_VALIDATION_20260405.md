# Eval Reporting E2E GitHub Actions Verification — Validation

日期：2026-04-05

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Qualifying `push/main` run found | **PASS** — `24066289833` |
| 2 | Run id / url / head sha / conclusion recorded | **PASS** |
| 3 | Evaluate job and deploy-pages job both succeeded | **PASS** |
| 4 | Consolidated deploy-pages summary step executed successfully | **PASS** |
| 5 | Required post-deploy eval reporting artifacts present | **PASS** |
| 6 | GitHub Pages enabled and workflow-backed URL recorded | **PASS** |
| 7 | External consumer behavior recorded truthfully | **PASS** |
| 8 | No code / workflow / test modifications in Batch 23 evidence collection | **PASS** |

## Verified Run

- Run ID: `24066289833`
- URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/24066289833`
- Event: `push`
- Branch: `main`
- Head SHA: `8d2dbb644f7c0a5e724217e3f41a8fff11594c90`
- Overall conclusion: `success`

## Job Conclusions

- `Run Evaluation and Generate Report` (`70192979470`): `success`
- `Deploy Report to GitHub Pages` (`70193123603`): `success`

Critical step confirmations:

- `Validate history with JSON Schema`: `success`
- `Consolidated eval reporting deploy-pages summary`: `success`
- `Deploy to GitHub Pages`: `success`

## Artifact / Pages Evidence

Artifact inventory for run `24066289833`:

- total_count = `15`
- required retained post-deploy surfaces all present:
  - `eval-reporting-public-index-1490`
  - `eval-reporting-dashboard-payload-1490`
  - `eval-reporting-webhook-delivery-request-1490`
  - `eval-reporting-webhook-delivery-result-1490`
  - `eval-reporting-release-draft-publish-result-1490`

Pages API:

- `html_url = https://zensgit.github.io/cad-ml-platform/`
- `build_type = workflow`
- `source.branch = main`
- `source.path = /`
- `public = true`
- `https_enforced = true`

## External Consumer Evidence

- `Send notifications`: `success`
- `Comment PR with results`: `skipped` on `push/main`, therefore N/A rather than failing
- `Post Eval Reporting status check`: step executed, but log records
  - `Status check skipped (fail-soft): Resource not accessible by integration`
- Commit status API for the same `head sha` returned `total_count = 0`

因此，本轮对 external consumer 的正确结论是：

- notifications surface: materialized
- PR comment surface: not applicable on this event
- commit status surface: fail-soft / permission-limited, 已如实记录

## Commands Run

```text
gh run view 24066289833 --json status,conclusion,event,headBranch,headSha,url,jobs
gh api repos/zensgit/cad-ml-platform/pages
gh api repos/zensgit/cad-ml-platform/actions/runs/24066289833/artifacts --paginate
gh api repos/zensgit/cad-ml-platform/actions/variables/HYBRID_SUPERPASS_FAIL_ON_FAILED
gh api repos/zensgit/cad-ml-platform/commits/8d2dbb644f7c0a5e724217e3f41a8fff11594c90/status
gh run view 24066289833 --job 70192979470 --log | rg -n "Post Eval Reporting status check|Status check skipped|Comment PR with results|Send notifications|Consolidated eval reporting deploy-pages summary"
```

## Validation Conclusion

Batch 23A 的真实 GitHub-hosted E2E verification 已完成并通过。

唯一未完全 materialize 的外部面是独立 commit status surface，但 workflow 以 fail-soft 方式记录了该权限限制，并未影响：

1. evaluate-side success
2. deploy-pages success
3. Pages 发布成功
4. post-deploy retained surfaces materialization

因此，Batch 23A 可以进入 closeout decision 阶段。

