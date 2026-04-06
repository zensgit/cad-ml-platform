# Eval Reporting Landing Page Refresh and Delivery Alignment — Design

日期：2026-03-30

## Scope

Batch 6B of the Claude Collaboration Eval Reporting plan:

- Integrate landing page into the refresh pipeline as step 4
- Add `landing_page_html` field to `eval_reporting_index.json`
- Upload landing page artifact in workflow
- Add `eval-reporting-landing-page` Makefile target

## Design Decisions

### 1. Refresh Pipeline — Step 4

`scripts/ci/refresh_eval_reporting_stack.py` now runs 4 steps:

1. `generate_eval_reporting_bundle` (fail-closed)
2. `check_eval_reporting_bundle_health` (fail-closed)
3. `generate_eval_reporting_index` (fail-closed)
4. `generate_eval_reporting_landing_page` (fail-closed)

The landing page step runs after the index so it can consume `eval_reporting_index.json`.

### 2. Index Additive Field

`eval_reporting_index.json` now includes:

```json
"landing_page_html": "<eval-history-dir>/index.html"
```

This is additive — no existing fields changed.

### 3. Workflow Integration

Two changes to `.github/workflows/evaluation-report.yml`:

1. `Upload eval reporting stack artifacts` step now includes `reports/eval_history/index.html`
2. New `Upload landing page` step uploads `reports/eval_history/index.html` as a dedicated artifact

### 4. Makefile Target

```makefile
eval-reporting-landing-page:
    $(PYTHON) scripts/generate_eval_reporting_landing_page.py \
        --eval-history-dir "$(EVAL_REPORTING_BUNDLE_EVAL_HISTORY_DIR)"
```

### 5. Owner Boundaries (unchanged)

The refresh script remains a pure orchestrator. The landing page renderer remains a pure consumer of existing artifacts.
