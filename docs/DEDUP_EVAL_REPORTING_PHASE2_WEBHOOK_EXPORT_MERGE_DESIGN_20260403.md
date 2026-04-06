# Eval Reporting Phase 2 Webhook Export Merge — Design

日期：2026-04-03

## Scope

Batch 18A: merge `webhook_export` into `delivery_request` by making `delivery_request` read `dashboard_payload` directly. No release chain changes.

## Changes

1. **`generate_eval_reporting_webhook_delivery_request.py`** — parameter renamed from `--webhook-export-json` to `--dashboard-payload-json`; reads `public_*` URL fields from dashboard_payload; `WEBHOOK_EVENT_TYPE` and `INGESTION_SCHEMA_VERSION` now constants in this file; `source_webhook_export_surface_kind` renamed to `source_dashboard_payload_surface_kind`

2. **Deleted:** `generate_eval_reporting_webhook_export.py`, `test_generate_eval_reporting_webhook_export.py`

3. **Workflow:** 3 webhook_export steps removed, sparse-checkout entry removed, delivery_request step input updated

## What Was NOT Changed

- `post_eval_reporting_webhook_delivery.js` — still reads `delivery_request.json` (unchanged contract)
- `eval_reporting_webhook_delivery_result.json` — unchanged schema
- Release chain — untouched
