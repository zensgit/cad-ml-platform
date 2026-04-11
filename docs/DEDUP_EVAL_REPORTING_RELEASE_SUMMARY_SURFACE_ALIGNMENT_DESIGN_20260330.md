# Eval Reporting Release Summary Surface Alignment — Design

日期：2026-03-30

## Scope

Batch 9A of the Claude Collaboration Eval Reporting plan:

- Create `scripts/ci/generate_eval_reporting_release_summary.py` thin helper
- Add three always-run workflow steps for release summary (generate, STEP_SUMMARY, upload)

## Design Decisions

### 1. Release Summary Helper

Reads existing `eval_reporting_index.json` and `eval_reporting_stack_summary.json`, normalizes into a release/status-friendly summary.

`eval_reporting_release_summary.json` fields:

| Field | Source |
|---|---|
| `status` | Same as `release_readiness` |
| `surface_kind` | `"eval_reporting_release_summary"` |
| `generated_at` | Current UTC |
| `stack_summary_status` | From stack summary |
| `missing_count` | From stack summary |
| `stale_count` | From stack summary |
| `mismatch_count` | From stack summary |
| `landing_page_path` | From index |
| `static_report_path` | From index |
| `interactive_report_path` | From index |
| `release_readiness` | Thin derived signal |

### 2. Release Readiness Signal

| Condition | readiness |
|---|---|
| Stack status == ok AND missing=0 AND stale=0 AND mismatch=0 | `ready` |
| Stack status present but any issue | `degraded` |
| No stack summary available | `unavailable` |

No new thresholds introduced — purely derived from existing health counts.

### 3. Workflow Steps

Three always-run steps inserted after stack summary upload, before fail step:

1. **Generate eval reporting release summary** — runs the helper
2. **Append eval reporting release summary to job summary** — cats MD to `$GITHUB_STEP_SUMMARY`
3. **Upload eval reporting release summary** — dedicated artifact

### 4. Owner Boundaries

Does NOT: regenerate bundles/health/index/public index, recompute metrics, render HTML.
