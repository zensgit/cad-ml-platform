# Eval Reporting Phase 4 Release Publish Payload Merge — Design

日期：2026-04-03

## Scope

Batch 20A: merge `publish_payload` into `publish_result` by making the JS consumer read `draft_payload` directly and derive publish policy internally. No draft_payload→publish_result deeper merge.

## Changes

1. **`post_eval_reporting_release_draft_publish.js`** — parameter renamed from `publishPayloadPath` to `draftPayloadPath`; `loadPublishPayload` renamed to `loadDraftPayload`; `publish_allowed` now derived internally from `readiness === "ready"`; `github_release_tag` generated internally

2. **Deleted:** `generate_eval_reporting_release_draft_publish_payload.py`, `test_generate_eval_reporting_release_draft_publish_payload.py`

3. **Workflow:** 3 publish_payload steps removed, sparse-checkout entry removed, publish_result step input updated to `draftPayloadPath`

## Release Chain (post-merge)

```
dashboard_payload → draft_payload → publish_result
```

## What Was NOT Changed

- `generate_eval_reporting_release_draft_payload.py` — still reads dashboard_payload
- `eval_reporting_release_draft_publish_result.json` — output schema/fields unchanged
- Webhook chain — untouched
