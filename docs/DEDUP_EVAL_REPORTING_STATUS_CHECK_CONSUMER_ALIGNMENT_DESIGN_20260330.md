# Eval Reporting Status Check Consumer Alignment — Design

日期：2026-03-30

## Scope

Batch 9B of the Claude Collaboration Eval Reporting plan:

- Create `scripts/ci/post_eval_reporting_status_check.js` thin status check consumer
- Add always-run status check step to workflow evaluate job

## Design Decisions

### 1. Status Check Consumer — `scripts/ci/post_eval_reporting_status_check.js`

Reads the release summary JSON and posts a GitHub commit status via the API.

Exported functions:
- `mapReadinessToState(readiness)` — maps to GitHub status API state
- `mapReadinessToDescription(readiness, summary)` — human-readable description
- `loadReleaseSummary(path)` — reads JSON, returns null if missing
- `postEvalReportingStatusCheck({github, context, releaseSummaryPath})` — posts the status

### 2. Status Mapping

| release_readiness | GitHub state | Description |
|---|---|---|
| `ready` | `success` | "Eval reporting stack is healthy" |
| `degraded` | `success` | "Degraded: missing=N, stale=N, mismatch=N" |
| `unavailable` | `failure` | "Eval reporting stack summary unavailable" |

Uses commit status API (`repos.createCommitStatus`) with context name `"Eval Reporting"`.

### 3. Fail-Soft Behavior

The step uses `continue-on-error: true`. If the API call fails (permissions, forks, etc.), it logs a warning and does not break the main workflow.

### 4. Workflow Placement

After release summary upload, before the fail step:

```
... Upload eval reporting release summary → Post Eval Reporting status check → Fail workflow on refresh failure ...
```

### 5. Owner Boundaries

The JS module does NOT: recompute release summary, read raw index/stack summary, generate comments/notifications/Pages content.
