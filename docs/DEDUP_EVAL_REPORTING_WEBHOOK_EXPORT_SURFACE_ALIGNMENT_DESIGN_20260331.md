# Eval Reporting Webhook Export Surface Alignment — Design

日期：2026-03-31

## Scope

Batch 11B: create `generate_eval_reporting_webhook_export.py` that reads only the dashboard payload and produces a stable webhook / ingestion export payload.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_dashboard_payload.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `stack_status`, `dashboard_headline`, `missing_count`, `stale_count`, `mismatch_count`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `webhook_event_type` (fixed: `eval_reporting.updated`), `ingestion_schema_version` (fixed: `1.0.0`).

MD contains: Webhook Event, Release readiness, Landing Page, Static Report, Interactive Report.

### 2. Workflow Steps (deploy-pages)

After release draft prefill upload: generate export, append to STEP_SUMMARY, upload as dedicated artifact. All always-run. Script added to sparse-checkout.

### 3. Owner Boundaries

Does NOT read release snippet, release summary, public index, or stack summary. Does NOT send HTTP requests.
