# Eval Reporting Stack PR Comment Consumer Alignment — Design

日期：2026-03-30

## Scope

Batch 7A of the Claude Collaboration Eval Reporting plan:

- Add `Eval Reporting Stack` section to PR comment
- Comment consumes existing `eval_reporting_stack_summary.json` and `eval_reporting_index.json`
- Workflow passes the two paths as env vars to the comment step

## Design Decisions

### 1. New JS Function — `summarizeEvalReportingStack`

Added to `scripts/ci/comment_evaluation_report_pr.js`:

```js
function summarizeEvalReportingStack(stackSummaryJsonPath, indexJsonPath) -> {
  available, status, light, summary,
  missingCount, staleCount, mismatchCount,
  landingPage, staticReport, interactiveReport
}
```

Reads the two JSON files. Returns a fallback view model when either is missing (available=false, light="⚪").

### 2. Comment Section — `Eval Reporting Stack`

A single row in a markdown table:

```
### Eval Reporting Stack
| Item | Value |
|---|---|
| **Status** | 🟢 status=ok, missing=0, stale=0, mismatch=0 |
```

Placed after the Strict Gate Decision Path section, before Quick Actions.

### 3. Workflow Env Vars

The PR comment step in `evaluation-report.yml` now includes:

```yaml
EVAL_REPORTING_STACK_SUMMARY_JSON_FOR_COMMENT: reports/ci/eval_reporting_stack_summary.json
EVAL_REPORTING_INDEX_JSON_FOR_COMMENT: reports/eval_history/eval_reporting_index.json
```

### 4. Missing Artifact Handling

When env vars are empty or files don't exist:
- `summarizeEvalReportingStack` returns `available: false`
- Comment still generates successfully
- Section shows "⚪ eval reporting stack summary not available"

### 5. Owner Boundaries

The comment script does NOT:
- Materialize bundles or health reports
- Recompute metrics or health checks
- Own any new summary schema
