# Eval Reporting Release Draft Dry Run Surface Alignment — Design

日期：2026-04-01

## Scope

Batch 12B: create `post_eval_reporting_release_draft_dry_run.js` that reads only the release-draft payload and produces a gated dry-run / optional-publish plan.

## Design Decisions

### 1. JS Module

Reads only `eval_reporting_release_draft_payload.json`. Produces JSON + MD dry-run plan.

Key functions:
- `buildDryRunPlan({payload, publishEnabled, tagPrefix})` — pure plan builder
- `postReleaseDraftDryRun({github, context, ...})` — workflow entry, writes artifacts, optionally creates draft release

### 2. Publish Gating

| publishEnabled | readiness | publish_mode | publish_allowed |
|---|---|---|---|
| false (default) | any | `dry_run` | false |
| true | `ready` | `publish` | true |
| true | other | `blocked` | false |

Default is always dry-run. No GitHub release created unless explicitly enabled AND readiness=ready.

### 3. Workflow Steps (deploy-pages)

After release draft payload upload: generate dry run (github-script), append to STEP_SUMMARY, upload artifact. All always-run + continue-on-error. JS file in sparse-checkout. `publishEnabled: false` hardcoded in default workflow.

### 4. Owner Boundaries

Does NOT read prefill, snippet, dashboard payload, or any upstream artifact. Does NOT auto-publish non-draft releases.
