# Eval Reporting Phase 5 Release Draft Payload Merge — Design

日期：2026-04-03

## Scope

Batch 21A: final release chain merge — absorb `draft_payload` into `publish_result` by making the JS consumer read `dashboard_payload` directly and derive draft title/body internally.

## Changes

1. **`post_eval_reporting_release_draft_publish.js`** — parameter renamed from `draftPayloadPath` to `dashboardPayloadPath`; `loadDraftPayload` renamed to `loadDashboardPayload`; draft_title/draft_body_markdown now derived internally from `public_*` URL fields and `release_readiness`/`dashboard_headline`

2. **Deleted:** `generate_eval_reporting_release_draft_payload.py`, `test_generate_eval_reporting_release_draft_payload.py`

3. **Workflow:** 3 draft_payload steps removed, sparse-checkout entry removed, publish_result step input updated to `dashboardPayloadPath`

## Final Release Chain

```
dashboard_payload → publish_result
```

Release chain depth is now 2 (was 6 at peak, before any merges).

## What Was NOT Changed

- `generate_eval_reporting_dashboard_payload.py` — unchanged
- `eval_reporting_release_draft_publish_result.json` — output schema/fields unchanged
- Webhook chain — untouched
