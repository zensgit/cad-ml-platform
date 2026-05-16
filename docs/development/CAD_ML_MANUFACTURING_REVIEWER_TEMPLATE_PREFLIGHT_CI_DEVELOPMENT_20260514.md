# CAD ML Manufacturing Reviewer Template Preflight CI Development

Date: 2026-05-14

## Goal

Run reviewer-template preflight automatically in the optional forward scorecard
flow before applying a filled reviewer template. This prevents incomplete or
duplicated template rows from being silently applied into the full manufacturing
review manifest.

## Changes

- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED`.
  - Runs `--validate-reviewer-template` when a filled
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV` exists.
  - Applies the filled template only when preflight status is `ready`.
  - Marks apply status as `blocked_preflight` when preflight is blocked.
  - Emits preflight outputs:
    - `manufacturing_reviewer_template_preflight_available`
    - `manufacturing_reviewer_template_preflight_summary_json`
    - `manufacturing_reviewer_template_preflight_status`
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for preflight summary and fail-on-blocked mode.
  - Uploads preflight summary as a dedicated artifact when available.
- Updated tests for:
  - no-template default output behavior
  - ready preflight before apply
  - blocked preflight fail-on-blocked behavior
  - workflow env and artifact upload wiring
- Updated Phase 6 TODO.

## CI Behavior

When a filled template is configured:

1. Run reviewer-template preflight.
2. If preflight is `ready`, apply the template into the review manifest.
3. If preflight is blocked, skip apply and validate the original review manifest.
4. If `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED`
   is true, fail the job after writing outputs and artifacts.

## Release Impact

The release workflow now has a guard before template apply. A malformed or
incomplete filled template produces a preflight summary and can block CI without
mutating the review manifest path used downstream.

## Remaining Work

- Produce the first real filled reviewer template from domain review.
- Run preflight in CI and resolve any reported rows.
- Apply, validate, merge, and tune thresholds once the reviewed set is stable.
