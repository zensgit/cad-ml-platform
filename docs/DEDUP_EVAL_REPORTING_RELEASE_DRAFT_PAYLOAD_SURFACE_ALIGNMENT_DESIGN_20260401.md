# Eval Reporting Release Draft Payload Surface Alignment — Design

日期：2026-04-01

## Scope

Batch 12A: create `generate_eval_reporting_release_draft_payload.py` that reads only the release-draft prefill and produces a GitHub release draft API-friendly payload.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_release_draft_prefill.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `draft_title`, `draft_body_markdown`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `repository_url`, `source_prefill_surface_kind`.

MD is the `draft_body_markdown` content.

### 2. Workflow Steps (deploy-pages)

After webhook export upload: generate payload, append to STEP_SUMMARY, upload as dedicated artifact. All always-run. Script added to sparse-checkout. Repository URL passed from `github.repository` context.

### 3. Owner Boundaries

Does NOT read release-note snippet, dashboard payload, release summary, public index, or stack summary. Does NOT create GitHub releases.
