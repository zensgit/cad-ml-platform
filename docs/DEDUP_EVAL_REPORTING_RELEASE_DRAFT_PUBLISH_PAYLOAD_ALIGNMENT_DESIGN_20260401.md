# Eval Reporting Release Draft Publish Payload Alignment — Design

日期：2026-04-01

## Scope

Batch 13A: create `generate_eval_reporting_release_draft_publish_payload.py` that reads only the release-draft payload and produces a publish-ready payload with explicit policy.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_release_draft_payload.json`. Outputs JSON + MD.

JSON fields: `status`, `surface_kind`, `generated_at`, `release_readiness`, `draft_title`, `draft_body_markdown`, `github_release_tag`, `publish_policy` (`disabled_by_default`), `publish_allowed` (true only when readiness=ready), `publish_requires_explicit_enable` (always true), `repository_url`, `landing_page_url`, `static_report_url`, `interactive_report_url`, `source_release_draft_payload_surface_kind`.

### 2. Workflow Steps (deploy-pages)

After dry-run upload: generate publish payload, append to STEP_SUMMARY, upload. All always-run. Script in sparse-checkout.

### 3. Owner Boundaries

Does NOT read upstream artifacts. Does NOT create GitHub releases.
