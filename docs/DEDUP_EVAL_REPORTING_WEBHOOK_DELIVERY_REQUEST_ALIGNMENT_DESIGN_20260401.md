# Eval Reporting Webhook Delivery Request Alignment — Design

日期：2026-04-01

## Scope

Batch 14A: create `generate_eval_reporting_webhook_delivery_request.py` that reads only the webhook export and produces a delivery request / policy artifact for external webhook senders.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_webhook_export.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `webhook_event_type`, `ingestion_schema_version`, `release_readiness`, `stack_status`, `dashboard_headline`, `missing_count`, `stale_count`, `mismatch_count`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `delivery_target_kind` (`external_webhook`), `delivery_method` (`POST`), `delivery_policy` (`disabled_by_default`), `delivery_allowed` (true when export exists), `delivery_requires_explicit_enable` (true), `request_timeout_seconds` (30), `request_body_json` (webhook export fields as JSON string), `source_webhook_export_surface_kind`.

### 2. Workflow Steps (deploy-pages)

After webhook export upload, before release draft payload: generate request, append to STEP_SUMMARY, upload. All always-run. Script in sparse-checkout.

### 3. Owner Boundaries

Does NOT read dashboard payload or upstream. Does NOT send HTTP requests or manage queues.
