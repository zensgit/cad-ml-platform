# Eval Reporting Dashboard Payload Surface Alignment — Design

日期：2026-03-31

## Scope

Batch 10A: create `generate_eval_reporting_dashboard_payload.py` thin helper and add three always-run steps to the deploy-pages job.

## Design Decisions

### 1. Dashboard Payload Helper

Reads `eval_reporting_release_summary.json` + `eval_reporting_public_index.json`, normalizes into an external-dashboard-friendly payload.

Fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `stack_status`, `missing_count`, `stale_count`, `mismatch_count`, `public_landing_page_url`, `public_static_report_url`, `public_interactive_report_url`, `dashboard_headline`, `public_discovery_ready`.

`dashboard_headline` and `public_discovery_ready` are thin derived signals from existing data.

### 2. Workflow Steps (deploy-pages)

After public index upload: download release summary artifact, generate dashboard payload, append to STEP_SUMMARY, upload as dedicated artifact. All always-run with continue-on-error.

### 3. Owner Boundaries

Does NOT regenerate release summary, public index, stack summary, bundles, health, or reports.
