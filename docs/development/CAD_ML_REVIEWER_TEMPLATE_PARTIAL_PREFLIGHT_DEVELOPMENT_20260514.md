# CAD ML Reviewer Template Partial Preflight Development

Date: 2026-05-14

## Goal

Support incremental manufacturing review batches without weakening release
validation. Batch reviewer templates may contain fewer rows than the release
minimum, so their preflight needs a separate minimum-ready-row threshold. The
full review manifest still uses the release minimum before it can be considered
release-label-ready.

## Changes

- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS`.
  - Reads it from
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS`.
  - Defaults to `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES`
    to preserve existing behavior when unset.
  - Uses the new value only for `--validate-reviewer-template`.
  - Keeps apply post-validation and manifest validation on the release minimum.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds env wiring for
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS`.
- Updated tests.
  - Adds a wrapper regression proving a one-row reviewer template can pass
    preflight with `PREFLIGHT_MIN_READY_ROWS=1` while the final manifest remains
    blocked against a release minimum of 30.
  - Adds workflow env coverage.
- Updated Phase 6 TODO.

## Configuration

Default behavior remains unchanged:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS
  defaults to FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES
```

For partial batch returns:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS=1
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES=30
```

This allows small approved reviewer batches to be applied, while the release
scorecard still reports the review manifest as blocked until the release
minimums are reached.

## Claude Code

Claude Code is available locally as `/Users/chouhua/.local/bin/claude`. It should
be used only for bounded, read-only assistance such as diff review or alternative
implementation review. Do not send secrets, tokens, private environment files,
or credentials to external model tools.

## Remaining Work

- Use the partial preflight threshold when reviewers return small batch templates.
- Continue applying approved batches until source, payload, and detail release
  minimums are met.
- Tune thresholds only after the reviewed release set is stable.
