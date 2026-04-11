# Eval Reporting Public Discovery Surface Alignment — Design

日期：2026-03-30

## Scope

Batch 8B of the Claude Collaboration Eval Reporting plan:

- Create `scripts/ci/generate_eval_reporting_public_index.py` thin public-surface helper
- Add public index generation, job summary append, and artifact upload steps to `deploy-pages` job

## Design Decisions

### 1. Public Index Helper — `scripts/ci/generate_eval_reporting_public_index.py`

Reads eval reporting index and stack summary, combines with Pages base URL, and outputs public-facing discovery JSON + MD.

`eval_reporting_public_index.json` fields:

| Field | Source |
|---|---|
| `status` | `"ok"` if page_url present, else `"no_page_url"` |
| `surface_kind` | `"eval_reporting_public_index"` |
| `generated_at` | Current UTC |
| `page_url` | From `--page-url` CLI arg (Pages deployment output) |
| `landing_page_url` | `<page_url>/index.html` |
| `static_report_url` | `<page_url>/report_static/index.html` |
| `interactive_report_url` | `<page_url>/report_interactive/index.html` |
| `stack_summary_status` | From stack summary JSON |
| `missing_count` | From stack summary JSON |
| `stale_count` | From stack summary JSON |
| `mismatch_count` | From stack summary JSON |

Does NOT: regenerate landing page, recompute health/bundle/index/summary.

### 2. deploy-pages Job Steps

After the Deploy step, four always-run steps are added:

1. **Checkout** — sparse checkout for the Python script
2. **Download artifacts** — stack + summary artifacts from evaluate job
3. **Generate public discovery index** — consumes `steps.deployment.outputs.page_url`
4. **Append public URLs to job summary** — cats MD to `$GITHUB_STEP_SUMMARY`
5. **Upload public discovery index** — dedicated artifact

All steps are `always()` / `continue-on-error: true` to avoid blocking Pages deployment.

### 3. Owner Boundaries

The public index helper is read-only — it only combines existing JSON + URL. No new metrics owner.
