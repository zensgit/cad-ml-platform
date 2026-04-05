# Eval Reporting Release Draft Prefill Surface Alignment — Design

日期：2026-03-31

## Scope

Batch 11A: create `generate_eval_reporting_release_draft_prefill.py` that reads only the release-note snippet and produces a draft title + body for GitHub release drafts or handoff.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_release_note_snippet.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `draft_title`, `draft_body_markdown`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `source_snippet_surface_kind`.

MD is the `draft_body_markdown` content, directly usable as a release body.

### 2. Workflow Steps (deploy-pages)

After snippet upload: generate prefill, append to STEP_SUMMARY, upload as dedicated artifact. All always-run. Script added to sparse-checkout.

### 3. Owner Boundaries

Does NOT read dashboard payload, release summary, public index, or stack summary. Only consumes snippet.
