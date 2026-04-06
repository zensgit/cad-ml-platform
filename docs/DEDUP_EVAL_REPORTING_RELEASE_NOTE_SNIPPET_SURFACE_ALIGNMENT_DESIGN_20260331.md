# Eval Reporting Release Note Snippet Surface Alignment — Design

日期：2026-03-31

## Scope

Batch 10B: create `generate_eval_reporting_release_note_snippet.py` that reads only the dashboard payload and produces a copy-pasteable snippet for release notes / handoff.

## Design Decisions

### 1. Snippet Helper

Reads only `eval_reporting_dashboard_payload.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `headline`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `snippet_markdown`.

MD contains: Release readiness, Landing Page link, Static Report link, Interactive Report link.

### 2. Workflow Steps (deploy-pages)

After dashboard payload upload: generate snippet, append to STEP_SUMMARY, upload as dedicated artifact. All always-run. Snippet script added to sparse-checkout.

### 3. Owner Boundaries

Does NOT read release summary, public index, or stack summary directly. Only consumes dashboard payload.
