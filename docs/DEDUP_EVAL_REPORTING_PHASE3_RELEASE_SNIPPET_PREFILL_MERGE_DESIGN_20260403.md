# Eval Reporting Phase 3 Release Snippet/Prefill Merge — Design

日期：2026-04-03

## Scope

Batch 19A: merge `release_note_snippet` and `release_draft_prefill` into `release_draft_payload` by making it read `dashboard_payload` directly. No publish_result deeper merge.

## Changes

1. **`generate_eval_reporting_release_draft_payload.py`** — parameter changed from `--prefill-json` to `--dashboard-payload-json`; now internally generates draft_title/draft_body_markdown from dashboard_payload fields; `source_prefill_surface_kind` renamed to `source_dashboard_payload_surface_kind`

2. **Deleted:** `generate_eval_reporting_release_note_snippet.py`, `generate_eval_reporting_release_draft_prefill.py`, and their tests

3. **Workflow:** 6 steps removed (snippet: 3, prefill: 3), 2 sparse-checkout entries removed, draft_payload step input updated

## What Was NOT Changed

- `generate_eval_reporting_release_draft_publish_payload.py` — still reads `release_draft_payload.json`
- `post_eval_reporting_release_draft_publish.js` — still reads `publish_payload.json`
- `eval_reporting_release_draft_publish_result.json` — unchanged schema
- Webhook chain — untouched
