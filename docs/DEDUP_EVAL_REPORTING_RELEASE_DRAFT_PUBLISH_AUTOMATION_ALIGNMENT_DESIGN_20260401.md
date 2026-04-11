# Eval Reporting Release Draft Publish Automation Alignment — Design

日期：2026-04-01

## Scope

Batch 13B: create `post_eval_reporting_release_draft_publish.js` that reads only the publish payload and executes a gated optional publish. Default is disabled.

## Design Decisions

### 1. JS Module

Reads only `eval_reporting_release_draft_publish_payload.json`. Produces result JSON + MD.

Key functions:
- `buildPublishResult({payload, publishEnabled})` — pure result builder
- `postReleaseDraftPublish({github, context, ...})` — workflow entry, optionally creates draft release

### 2. Publish Gating

| publishEnabled | publish_allowed (from payload) | publish_mode |
|---|---|---|
| false (default) | any | `disabled` |
| true | true | `publish` |
| true | false | `blocked` |

Default: `publishEnabled: false` hardcoded in workflow. No GitHub release created.

### 3. Workflow Steps (deploy-pages)

After publish payload upload: generate result (github-script), append to STEP_SUMMARY, upload. All always-run + continue-on-error. JS in sparse-checkout.

### 4. Owner Boundaries

Does NOT read upstream artifacts. Only creates draft releases (never non-draft). Fail-soft on API errors.
